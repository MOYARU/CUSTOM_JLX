#pragma once
//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
// HiMA — Hierarchical Memory-Aware Training
//
// Hot  (Metal buffer) → 정밀 학습, 매 step
// Warm (CPU mmap)     → 느슨한 학습, 매 K step
// Cold (SSD file)     → frozen, forward만 참여
//
// Gradient magnitude로 동적 승격/강등
//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <fstream>


enum class Tier { HOT = 0, WARM = 1, COLD = 2 };

struct BlockInfo {
    int layer_idx;          // 어떤 layer의
    int proj_idx;           // 어떤 projection (0=Q,1=K,2=V,3=O)
    Tier tier;              // 현재 tier
    float grad_score;       // EMA of |gradient| — 승격/강등 기준
    float grad_accum;       // Warm tier용 gradient 누적량
    int steps_since_update; // 마지막 update 이후 step 수
    int steps_in_tier;      // 현재 tier에 머문 step 수

    // 간단한 이름
    std::string name() const {
        static const char* pnames[] = {"Q","K","V","O","gate","up","down"};
        return "L" + std::to_string(layer_idx) + "_" +
               (proj_idx < 7 ? pnames[proj_idx] : "?");
    }
};

struct HiMAConfig {
    // Tier 비율 (합 = 1.0)
    float hot_ratio  = 0.40f;  // 40% of blocks → Hot
    float warm_ratio = 0.35f;  // 35% → Warm
    // 나머지 25% → Cold

    // 학습 전략
    int warm_update_interval = 10;   // Warm은 매 10 step마다 update
    int rebalance_interval   = 50;   // 매 50 step마다 리밸런싱
    int cold_probe_count     = 4;    // 리밸런싱 시 Cold에서 몇 개를 probe할지

    // Gradient score EMA
    float grad_ema_alpha = 0.1f;     // score = alpha * new + (1-alpha) * old

    // 승격/강등 hysteresis (너무 자주 이동 방지)
    int min_steps_in_tier = 20;      // 최소 20 step은 현재 tier에 머물기

    // Warm tier 학습률 배
    float warm_lr_scale = 0.1f;      // Hot의 10%

    // Cold tier: forward에서 stale cache 사용
    bool cold_use_stale_cache = true;
};

class TierManager {
public:
    TierManager(int num_blocks, const HiMAConfig& cfg = HiMAConfig());

    // 초기 tier 할당 (균등 분배)
    void initialize();

    // 매 step 호출: gradient magnitude 기록
    void record_gradient(int block_idx, float grad_magnitude);

    // 이 block을 이번 step에서 update해야 하는가?
    bool should_update(int block_idx, int current_step) const;

    // 이 block의 학습률 배율
    float lr_scale(int block_idx) const;

    // 이 block이 backward에 참여하는가?
    bool needs_backward(int block_idx) const;

    // 리밸런싱 실행 (매 R step마다 main에서 호출)
    // 반환: 이동된 block 수
    int rebalance(int current_step);

    // 접근자
    Tier get_tier(int block_idx) const { return blocks_[block_idx].tier; }
    const BlockInfo& get_block(int block_idx) const { return blocks_[block_idx]; }
    int num_blocks() const { return (int)blocks_.size(); }

    int count_hot()  const;
    int count_warm() const;
    int count_cold() const;

    // 상태 출력
    void print_status() const;

    // 전체 step 카운터 증가
    void step() { global_step_++; }
    int current_step() const { return global_step_; }

private:
    std::vector<BlockInfo> blocks_;
    HiMAConfig cfg_;
    int global_step_ = 0;
    int num_hot_, num_warm_, num_cold_;

    void assign_tiers_by_score();
};
