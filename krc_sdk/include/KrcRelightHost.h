/*
 * KRC: CPU side of exact relighting (KrcRelight.h). Engine-independent, C++17, no dependency.
 *
 * Each frame, the engine gives the power of each lamp (intensity × luminance of its colour)
 * and its group (0..2 tracked, 3 fixed); KrcRelightHost computes the "after / before" ratio of each
 * group. A group is unknown (ratio 1, not applied) if its lamps do not all change by the same
 * ratio or if one of them turns on from zero: its share in the cells then says nothing about the
 * new light.
 *
 *   KrcRelightHost host;
 *   host.BeginFrame(cacheCleared);
 *   for (lamp i) host.AddLight(i, group, intensity * luminance(color));
 *   KrcRelightFrame f = host.EndFrame();
 *   // f.ratio, f.lightGroups -> KrcRelightParameters; f.event -> KrcRelightEntry pass before the
 *   // SHaRC update; f.active -> S differs from 1 somewhere (denoiser)
 */

#ifndef KRC_RELIGHT_HOST_H
#define KRC_RELIGHT_HOST_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct KrcRelightFrame
{
    float ratio[3] = { 1.0f, 1.0f, 1.0f };
    uint32_t lightGroups = 0; // group of lamp i in bits 2i..2i+1
    bool event = false;       // a known ratio must be applied this frame
    bool active = false;      // a relighting has happened since the cache was reset
};

class KrcRelightHost
{
public:
    void BeginFrame(bool cacheCleared)
    {
        m_frame = KrcRelightFrame();
        for (int g = 0; g < 3; g++)
        {
            m_seen[g] = false;
            m_known[g] = true;
        }
        if (cacheCleared)
            m_active = false;
    }

    // lightIndex: stable index of the lamp from one frame to the next (16 at most for lightGroups)
    void AddLight(uint32_t lightIndex, int group, float power)
    {
        group = std::min(std::max(group, 0), 3);
        if (lightIndex < 16)
            m_frame.lightGroups |= uint32_t(group) << (2 * lightIndex);
        if (m_prevPower.size() <= lightIndex)
            m_prevPower.resize(lightIndex + 1, -1.0f);

        const float prev = m_prevPower[lightIndex] < 0.0f ? power : m_prevPower[lightIndex];
        m_prevPower[lightIndex] = power;
        if (group == 3 || (prev <= 0.0f && power <= 0.0f))
            return;
        if (prev <= 0.0f)
        {
            m_known[group] = false; // turns on from zero: unknown ratio
            return;
        }
        const float r = power / prev;
        if (!m_seen[group])
        {
            m_frame.ratio[group] = r;
            m_seen[group] = true;
        }
        else if (std::abs(r - m_frame.ratio[group]) > 1e-4f * std::max(r, m_frame.ratio[group]))
            m_known[group] = false;
    }

    KrcRelightFrame EndFrame()
    {
        for (int g = 0; g < 3; g++)
        {
            if (!m_known[g])
                m_frame.ratio[g] = 1.0f;
            m_frame.event = m_frame.event || m_frame.ratio[g] != 1.0f;
        }
        m_active = m_active || m_frame.event;
        m_frame.active = m_active;
        return m_frame;
    }

private:
    KrcRelightFrame m_frame;
    std::vector<float> m_prevPower;
    bool m_seen[3] = {};
    bool m_known[3] = {};
    bool m_active = false;
};

#endif // KRC_RELIGHT_HOST_H
