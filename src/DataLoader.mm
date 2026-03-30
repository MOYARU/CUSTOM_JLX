#include "DataLoader.h"
#include <random>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <regex>

namespace fs = std::filesystem;

// single file
DataLoader::DataLoader(const std::string& filepath, int batch_size, int seq_len)
    : batch_size_(batch_size), seq_len_(seq_len)
{
    shards_.push_back(filepath);
    open_shard(0);
    load_next_chunk();

    // 전체 토큰 수 추정 (파일 크기 기준)
    total_tok_ = 0;
    for (auto& s : shards_) {
        auto sz = fs::file_size(s);
        total_tok_ += sz / sizeof(uint16_t);
    }
    std::cout << "DataLoader: " << total_tok_ << " tokens"
              << " (" << shards_.size() << " shard(s))\n";
}

// shard directory
DataLoader::DataLoader(const std::string& dir, const std::string& pattern,
                       int batch_size, int seq_len)
    : batch_size_(batch_size), seq_len_(seq_len)
{
    // pattern 예: "shard_*.bin"
    std::vector<std::string> found;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        // 간단한 glob: * 를 정규식
        std::string pat = pattern;
        std::string re_str;
        for (char c : pat) {
            if (c == '*') re_str += ".*";
            else if (c == '?') re_str += ".";
            else if (std::string("^$.|+()[]{}\\").find(c) != std::string::npos)
                re_str += std::string("\\") + c;
            else re_str += c;
        }
        if (std::regex_match(name, std::regex(re_str)))
            found.push_back(e.path().string());
    }
    std::sort(found.begin(), found.end());
    shards_ = found;

    if (shards_.empty()) {
        std::cerr << "DataLoader: no shards found in " << dir
                  << " matching " << pattern << "\n";
        return;
    }

    open_shard(0);
    load_next_chunk();

    total_tok_ = 0;
    for (auto& s : shards_) total_tok_ += fs::file_size(s) / sizeof(uint16_t);

    std::cout << "DataLoader: " << total_tok_/1e9 << "B tokens"
              << " across " << shards_.size() << " shards\n";
}

// open shard
void DataLoader::open_shard(int idx) {
    cur_file_.close();
    cur_file_.open(shards_[idx], std::ios::binary);
    if (!cur_file_) {
        std::cerr << "DataLoader: cannot open " << shards_[idx] << "\n";
        exit(1);  // 파일 없으면 즉시 종료 (무한루프 방지)
    }
    shard_tokens_remaining_ = fs::file_size(shards_[idx]) / sizeof(uint16_t);
    cur_shard_ = idx;
}

// load next chunk
void DataLoader::load_next_chunk() {
    size_t need = CHUNK_TOKENS;

    // seq_len+1 경계를 맞추기 위해 실제 필요량 계산
    // 배치 하나당 batch_size * seq_len + 1 토큰 필요
    size_t batch_tokens = (size_t)batch_size_ * seq_len_ + 1;
    need = std::max(need, batch_tokens * 4);  // 최소 4배치 분량

    chunk_.clear();
    chunk_.reserve(need + 1);

    while (chunk_.size() < need) {
        size_t can_read = std::min(need - chunk_.size(), shard_tokens_remaining_);
        if (can_read > 0) {
            size_t old_size = chunk_.size();
            chunk_.resize(old_size + can_read);
            cur_file_.read((char*)(chunk_.data() + old_size),
                          can_read * sizeof(uint16_t));
            size_t actually_read = cur_file_.gcount() / sizeof(uint16_t);
            chunk_.resize(old_size + actually_read);
            shard_tokens_remaining_ -= actually_read;
        }

        if (shard_tokens_remaining_ == 0 || !cur_file_) {
            // 다음 shard로
            int next = cur_shard_ + 1;
            if (next >= (int)shards_.size()) {
                // epoch 완료, 처음으로
                next = 0;
                epoch_++;
                std::cout << "[epoch " << epoch_ << "]\n" << std::flush;
            }
            open_shard(next);
        }
    }

    // chunk 내 sequence 단위 셔플 (같은 배치가 반복되지 않도록)
    size_t seq_tokens = (size_t)batch_size_ * seq_len_ + 1;
    size_t n_seqs = chunk_.size() / seq_tokens;
    if (n_seqs > 1) {
        // Fisher-Yates shuffle over seq-length blocks
        static std::mt19937 rng(std::random_device{}());
        for (size_t i = n_seqs - 1; i > 0; --i) {
            size_t j = std::uniform_int_distribution<size_t>(0, i)(rng);
            if (i != j) {
                std::swap_ranges(
                    chunk_.begin() + i * seq_tokens,
                    chunk_.begin() + (i+1) * seq_tokens,
                    chunk_.begin() + j * seq_tokens);
            }
        }
    }

    chunk_pos_ = 0;
}

// get_batch
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DataLoader::get_batch()
{
    size_t need = (size_t)batch_size_ * seq_len_ + 1;

    // 청크가 부족하면 다음 청크 로드
    if (chunk_pos_ + need > chunk_.size()) {
        load_next_chunk();
    }

    auto input  = std::make_shared<Tensor>(
        std::vector<int>{batch_size_, seq_len_}, DType::Int32);
    auto target = std::make_shared<Tensor>(
        std::vector<int>{batch_size_, seq_len_}, DType::Int32);

    int32_t* in_p = (int32_t*)input->data();
    int32_t* tg_p = (int32_t*)target->data();

    for (int b = 0; b < batch_size_; ++b) {
        for (int s = 0; s < seq_len_; ++s) {
            size_t idx = chunk_pos_ + (size_t)b * seq_len_ + s;
            in_p[b * seq_len_ + s] = (int32_t)chunk_[idx];
            tg_p[b * seq_len_ + s] = (int32_t)chunk_[idx + 1];
        }
    }

    chunk_pos_ += (size_t)batch_size_ * seq_len_;
    tokens_seen_ += (size_t)batch_size_ * seq_len_;
    return {input, target};
}