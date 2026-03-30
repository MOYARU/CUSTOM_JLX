#include "TierManager.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

// HiMA 논문은 나중에 공개

TierManager::TierManager(int num_blocks, const HiMAConfig& cfg)
    : cfg_(cfg)
{
    if (num_blocks <= 0) {
        num_hot_ = 0; num_warm_ = 0; num_cold_ = 0;
        std::cout << "HiMA WARNING: 0 blocks registered\n"; // warning
        return;
    }

    blocks_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks_[i].layer_idx = i / 7;  // 7 projections per layer (Q,K,V,O + gate,up,down)
        blocks_[i].proj_idx  = i % 7;
        blocks_[i].tier = Tier::HOT;
        blocks_[i].grad_score = 0.f;
        blocks_[i].grad_accum = 0.f;
        blocks_[i].steps_since_update = 0;
        blocks_[i].steps_in_tier = 0;
    }

    num_hot_  = std::max(1, (int)(num_blocks * cfg_.hot_ratio));
    num_warm_ = std::max(1, (int)(num_blocks * cfg_.warm_ratio));
    num_cold_ = num_blocks - num_hot_ - num_warm_;
    if (num_cold_ < 0) { num_cold_ = 0; num_warm_ = num_blocks - num_hot_; }

    initialize();
}

void TierManager::initialize() {
    // 처음에는 앞쪽 layer를 Hot, 뒤쪽을 Cold로 사용
    std::vector<int> indices(blocks_.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 역순: 뒤쪽 layer (LM head에 가까운) -> 높은 우선순위
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return blocks_[a].layer_idx > blocks_[b].layer_idx;
    });

    for (int rank = 0; rank < (int)indices.size(); ++rank) {
        int idx = indices[rank];
        if (rank < num_hot_)
            blocks_[idx].tier = Tier::HOT;
        else if (rank < num_hot_ + num_warm_)
            blocks_[idx].tier = Tier::WARM;
        else
            blocks_[idx].tier = Tier::COLD;
    }

    std::cout << "[HiMA] Initialized: "
              << num_hot_ << " Hot, "
              << num_warm_ << " Warm, "
              << num_cold_ << " Cold ("
              << blocks_.size() << " total blocks)\n";
}

void TierManager::record_gradient(int block_idx, float grad_magnitude) {
    auto& b = blocks_[block_idx];
    // EMA update
    if (b.grad_score == 0.f)
        b.grad_score = grad_magnitude;
    else
        b.grad_score = cfg_.grad_ema_alpha * grad_magnitude
                     + (1.f - cfg_.grad_ema_alpha) * b.grad_score;

    // Warm tier: accumulate
    if (b.tier == Tier::WARM)
        b.grad_accum += grad_magnitude;

    b.steps_since_update++;
    b.steps_in_tier++;
}

bool TierManager::should_update(int block_idx, int /*current_step*/) const {
    const auto& b = blocks_[block_idx];
    switch (b.tier) {
        case Tier::HOT:
            return true;  // 매 step
        case Tier::WARM:
            return (b.steps_since_update >= cfg_.warm_update_interval);
        case Tier::COLD:
            return false;  // update X
    }
    return false;
}

float TierManager::lr_scale(int block_idx) const {
    switch (blocks_[block_idx].tier) {
        case Tier::HOT:  return 1.0f;
        case Tier::WARM: return cfg_.warm_lr_scale;
        case Tier::COLD: return 0.0f;
    }
    return 0.f;
}

bool TierManager::needs_backward(int block_idx) const {
    // Hot, Warm: backward 참여 (weight gradient 필요)
    // Cold: forward만 (gradient 계산 skip)
    return blocks_[block_idx].tier != Tier::COLD;
}

int TierManager::rebalance(int /*current_step*/) {
    // 모든 block을 gradient score로 정렬
    std::vector<int> indices(blocks_.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return blocks_[a].grad_score > blocks_[b].grad_score;
    });

    int moves = 0;

    for (int rank = 0; rank < (int)indices.size(); ++rank) {
        int idx = indices[rank];
        auto& b = blocks_[idx];

        // Hysteresis: 최소 체류 시간 확인
        if (b.steps_in_tier < cfg_.min_steps_in_tier)
            continue;

        Tier new_tier;
        if (rank < num_hot_)
            new_tier = Tier::HOT;
        else if (rank < num_hot_ + num_warm_)
            new_tier = Tier::WARM;
        else
            new_tier = Tier::COLD;

        if (new_tier != b.tier) {
            Tier old_tier = b.tier;
            b.tier = new_tier;
            b.steps_in_tier = 0;
            b.steps_since_update = 0;
            b.grad_accum = 0.f;
            moves++;

            // 승격은 축하, 강등은 조용히
            if (new_tier < old_tier) {  // 승격
                // 승격 로그는 rebalance 요약에서 출력
            }
        }
    }

    return moves;
}

int TierManager::count_hot()  const {
    int c=0; for(auto& b:blocks_) if(b.tier==Tier::HOT) c++; return c;
}
int TierManager::count_warm() const {
    int c=0; for(auto& b:blocks_) if(b.tier==Tier::WARM) c++; return c;
}
int TierManager::count_cold() const {
    int c=0; for(auto& b:blocks_) if(b.tier==Tier::COLD) c++; return c;
}

void TierManager::print_status() const {
    std::cout << "[HiMA] Step " << global_step_
              << " | Hot:" << count_hot()
              << " Warm:" << count_warm()
              << " Cold:" << count_cold();

    // Top-3 blocks by gradient score (any tier)
    std::vector<int> all_idx(blocks_.size());
    std::iota(all_idx.begin(), all_idx.end(), 0);
    std::sort(all_idx.begin(), all_idx.end(), [&](int a,int b){
        return blocks_[a].grad_score > blocks_[b].grad_score; });

    std::cout << " | Top: ";
    int shown = 0;
    for (int idx : all_idx) {
        if (shown >= 3) break;
        auto& b = blocks_[idx];
        if (b.grad_score <= 0.f) continue;
        const char* tier_name[] = {"H","W","C"};
        std::cout << b.name() << "(" << tier_name[(int)b.tier]
                  << ",g=" << std::scientific << std::setprecision(1)
                  << b.grad_score << ") ";
        shown++;
    }
    if (shown == 0) std::cout << "(no gradients yet)";
    std::cout << std::fixed << "\n";
}
