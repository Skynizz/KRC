/*
 * KRC: exact relighting by lamp groups, as an extension of SHaRC (v1.8.x).
 *
 * Engine-independent: buffers and settings go through KrcRelightParameters, like SHaRC
 * with SharcParameters. Include after SharcCommon.h. How to use it: docs/INTEGRATION.md.
 *
 * Principle. Each SHaRC cell learns the share of its light that comes from each lamp group
 * (3 tracked groups; the rest, sky and emissives, is fixed). The direct light of a vertex
 * is stored in the group of the sampled lamp and flows back to the previous cells with SHaRC's
 * weights; a vertex that reads the cache brings its cell's shares. When the engine
 * changes a group's intensity by a known ratio r ("after / before"), KrcRelightEntry
 * multiplies each cell's radiance by k = 1 + Σ f_g (r_g − 1) before the frame is traced:
 * the cache jumps to the new lighting at once, with no noise and no lag. Exact for luminance; the
 * colour is approximate if the groups do not share the same tint.
 *
 * The product S of each cell's factors k serves the denoiser: its input is divided by S
 * (smoothed on screen) and its output multiplied back by S, so its history comes out relit even
 * though it has no access to it (NRD, DLSS RR…).
 *
 * Buffers, one entry per entry of SHaRC's table, reset along with it:
 *   groupAccumulationBuffer  uint4  fixed-point sums of the frame (xyz groups, w total)
 *   groupBuffer              float4 smoothed means (same layout)
 *   factorBuffer             float2 x: factor of the frame, y: product S (0 means 1)
 */

#ifndef KRC_RELIGHT_H
#define KRC_RELIGHT_H

#ifndef KRC_GROUP_SCALE
#define KRC_GROUP_SCALE 1e3f // fixed point of the sums (like SHARC_RADIANCE_SCALE)
#endif

struct KrcRelightParameters
{
    RW_STRUCTURED_BUFFER(groupAccumulationBuffer, uint4);
    RW_STRUCTURED_BUFFER(groupBuffer, float4);
    RW_STRUCTURED_BUFFER(factorBuffer, float2);
    uint lightGroups;   // group of lamp i in bits 2i..2i+1 (0..2 tracked, 3 fixed); 16 lamps at most
    float3 ratio;       // "after / before" ratio of groups 0..2 (1: unchanged or unknown)
};

uint KrcLightGroup(KrcRelightParameters krcParameters, uint lightIndex)
{
    return lightIndex < 16u ? (krcParameters.lightGroups >> (2u * lightIndex)) & 3u : 3u;
}

// direct light of a lamp, stored in its group (luminance); to be summed over the vertex's lamps
float3 KrcGroupSplit(KrcRelightParameters krcParameters, uint lightIndex, float3 lightContribution)
{
    const uint g = KrcLightGroup(krcParameters, lightIndex);
    const float lum = SharcLuma(lightContribution);
    return float3(g == 0u ? lum : 0.0f, g == 1u ? lum : 0.0f, g == 2u ? lum : 0.0f);
}

#if SHARC_UPDATE
void KrcGroupAdd(KrcRelightParameters krcParameters, HashGridIndex cell, float4 value)
{
    const uint4 s = uint4(max(value, 0.0f) * KRC_GROUP_SCALE);
    if (s.x != 0) InterlockedAdd(BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, cell).x, s.x);
    if (s.y != 0) InterlockedAdd(BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, cell).y, s.y);
    if (s.z != 0) InterlockedAdd(BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, cell).z, s.z);
    if (s.w != 0) InterlockedAdd(BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, cell).w, s.w);
}

// Just before SharcUpdateHit, with the same state, the same direct light and the same random
// number: redoes its insertion and its cache-read decision, and stores the shares in the
// same cells. directGroups: sum of the KrcGroupSplit of the vertex's lamps.
void KrcUpdateHit(in SharcParameters sharcParameters, KrcRelightParameters krcParameters, in SharcState sharcState, SharcHitData sharcHitData,
    float3 directLighting, float3 directGroups, float random)
{
    HashGridKey hashGridKey;
    HashGridIndex cell;
    if (!HashGridInsertEntry(sharcParameters.hashGridData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters, hashGridKey, cell))
        return;

    float3 materialDemodulation = float3(1.0f, 1.0f, 1.0f);
#if SHARC_MATERIAL_DEMODULATION
    materialDemodulation = sharcHitData.materialDemodulation;
#endif

    float3 radiance = directLighting;
    float4 groups = float4(directGroups, SharcLuma(directLighting));
    bool resampled = false;
#if SHARC_ENABLE_CACHE_RESAMPLING
    const uint resamplingDepth = uint(round(lerp(SHARC_RESAMPLING_DEPTH_MIN, SHARC_PROPAGATION_DEPTH, random)));
    if (resamplingDepth <= sharcState.pathLength)
    {
        const SharcVoxelData voxelData = SharcGetVoxelData(sharcParameters.resolvedBuffer, cell);
        if (voxelData.accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
        {
            radiance = SharcDecodeRadiance(voxelData.accumulatedRadiance, SharcGetRadianceDirection(sharcHitData)) * materialDemodulation;
            const float4 f = BUFFER_AT_OFFSET(krcParameters.groupBuffer, cell);
            const float lum = SharcLuma(radiance);
            groups = float4(f.w > 0.0f ? saturate(f.xyz / f.w) * lum : float3(0, 0, 0), lum);
            resampled = true;
        }
    }
#endif // SHARC_ENABLE_CACHE_RESAMPLING

    const float lum = SharcLuma(radiance);
    if (lum <= 0.0f)
        return;

    // the vertex's cell keeps its demodulated direct light (like SHaRC), unless the cache is read
    if (!resampled)
        KrcGroupAdd(krcParameters, cell, groups * (SharcLuma(radiance / materialDemodulation) / lum));

    for (uint i = 0; i < sharcState.pathLength; ++i)
    {
        const HashGridIndex previous = sharcState.cacheIndices[i];
        if (previous == HASH_GRID_INVALID_CACHE_INDEX)
            continue;
        KrcGroupAdd(krcParameters, previous, groups * (SharcLuma(radiance * float3(sharcState.sampleWeights[i])) / lum));
    }
}

// just before SharcUpdateMiss: the sky only counts in the total
void KrcUpdateMiss(KrcRelightParameters krcParameters, in SharcState sharcState, float3 radiance)
{
    for (uint i = 0; i < sharcState.pathLength; ++i)
    {
        const HashGridIndex previous = sharcState.cacheIndices[i];
        if (previous == HASH_GRID_INVALID_CACHE_INDEX)
            continue;
        KrcGroupAdd(krcParameters, previous, float4(0, 0, 0, SharcLuma(radiance * float3(sharcState.sampleWeights[i]))));
    }
}
#endif // SHARC_UPDATE

// In the resolve pass, just before SharcResolveEntry: mean of the shares with SHaRC's weights
// (accumulationFrameNum: the one from SharcResolveParameters). Returns whether the entry was occupied,
// to pass to KrcResolveEviction. Empty entries (most of the table) cost only one read.
bool KrcResolveEntry(uint entryIndex, SharcParameters sharcParameters, KrcRelightParameters krcParameters, uint accumulationFrameNum)
{
    if (entryIndex >= sharcParameters.hashGridData.capacity)
        return false;
    if (BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, entryIndex) == HASH_GRID_INVALID_HASH_KEY)
        return false;

    const uint4 acc = BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, entryIndex);
    if (any(acc != 0))
        BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, entryIndex) = uint4(0, 0, 0, 0);

    const float sampleNum = SharcGetAccumulatedSampleNum(BUFFER_AT_OFFSET(sharcParameters.accumulationBuffer, entryIndex));
    if (sampleNum == 0)
        return true;

    const SharcVoxelData voxelData = SharcGetVoxelData(sharcParameters.resolvedBuffer, entryIndex);
    float sampleNumPrev = voxelData.accumulatedSampleNum;
    const uint accumulatedFrameNum = voxelData.accumulatedFrameNum + 1;
    const uint frameNumMax = clamp(accumulationFrameNum, SHARC_ACCUMULATED_FRAME_NUM_MIN, SHARC_ACCUMULATED_FRAME_NUM_MAX);
    if (accumulatedFrameNum > frameNumMax)
        sampleNumPrev *= float(frameNumMax) / float(accumulatedFrameNum);

    const float4 frameMean = float4(acc) / (KRC_GROUP_SCALE * sampleNum);
    BUFFER_AT_OFFSET(krcParameters.groupBuffer, entryIndex) =
        (BUFFER_AT_OFFSET(krcParameters.groupBuffer, entryIndex) * sampleNumPrev + frameMean * sampleNum) / (sampleNumPrev + sampleNum);
    return true;
}

// Right after SharcResolveEntry: a cell that SHaRC has just evicted frees its shares and its
// product S, so that the next cell stored there starts again from zero (and from S = 1).
void KrcResolveEviction(uint entryIndex, SharcParameters sharcParameters, KrcRelightParameters krcParameters, bool wasValid)
{
    if (!wasValid || BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, entryIndex) != HASH_GRID_INVALID_HASH_KEY)
        return;
    BUFFER_AT_OFFSET(krcParameters.groupAccumulationBuffer, entryIndex) = uint4(0, 0, 0, 0);
    BUFFER_AT_OFFSET(krcParameters.groupBuffer, entryIndex) = float4(0, 0, 0, 0);
    BUFFER_AT_OFFSET(krcParameters.factorBuffer, entryIndex) = float2(0, 0);
}

// Separate pass, one entry per thread, before SHaRC's update on a frame where a ratio is
// known: relights the cell and records its factor.
void KrcRelightEntry(uint entryIndex, SharcParameters sharcParameters, KrcRelightParameters krcParameters)
{
    if (entryIndex >= sharcParameters.hashGridData.capacity)
        return;
    if (BUFFER_AT_OFFSET(sharcParameters.hashGridData.hashEntriesBuffer, entryIndex) == HASH_GRID_INVALID_HASH_KEY)
        return;
    float2 factor = BUFFER_AT_OFFSET(krcParameters.factorBuffer, entryIndex);

    const float4 groups = BUFFER_AT_OFFSET(krcParameters.groupBuffer, entryIndex);
    if (groups.w <= 0.0f)
    {
        BUFFER_AT_OFFSET(krcParameters.factorBuffer, entryIndex) = float2(1.0f, factor.y);
        return;
    }

    const float3 f = saturate(groups.xyz / groups.w);
    const float k = max(1.0f + dot(f, krcParameters.ratio - 1.0f), 0.0f);

    SharcVoxelData voxelData = SharcGetVoxelData(sharcParameters.resolvedBuffer, entryIndex);
    BUFFER_AT_OFFSET(sharcParameters.resolvedBuffer, entryIndex) = SharcPackVoxelData(SharcScaleRadianceData(voxelData.accumulatedRadiance, k),
        voxelData.accumulatedSampleNum, voxelData.accumulatedFrameNum, voxelData.staleFrameNum, voxelData.sampleDataExt);
    BUFFER_AT_OFFSET(krcParameters.groupBuffer, entryIndex) = float4(groups.xyz * krcParameters.ratio, groups.w * k);
    BUFFER_AT_OFFSET(krcParameters.factorBuffer, entryIndex) = float2(k, (factor.y == 0.0f ? 1.0f : factor.y) * k);
}

// Factors of a point's cell (x: of the frame, y: product S); (1, 1) without a cell. At the
// first hit point, to be written to a texture for the denoiser.
float2 KrcGetRelightFactor(in SharcParameters sharcParameters, KrcRelightParameters krcParameters, float3 position, float3 normal)
{
    HashGridKey hashGridKey;
    const HashGridIndex cell = HashGridFindEntry(sharcParameters.hashGridData, position, normal, sharcParameters.hashGridParameters, hashGridKey);
    if (cell == HASH_GRID_INVALID_CACHE_INDEX)
        return float2(1.0f, 1.0f);
    const float2 factor = BUFFER_AT_OFFSET(krcParameters.factorBuffer, cell);
    return float2(factor.x, factor.y == 0.0f ? 1.0f : factor.y);
}

#endif // KRC_RELIGHT_H
