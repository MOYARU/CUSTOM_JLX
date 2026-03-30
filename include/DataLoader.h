#pragma once

#include "Tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <cstdint>

// 메모리 맵 방식 DataLoader
// - 전체 데이터를 RAM에 올리지 않음
// - 청크 단위로 읽어서 스트리밍
// - 수십억 토큰 파일 지원
// - 데이터 포맷: uint16 토큰 ID 연속 저장 (.bin)
// -  또는 uint32 토큰 ID (.bin32)
// - 여러 파일 shard 지원

class DataLoader {
public:
    // single file
    DataLoader(const std::string& filepath, int batch_size, int seq_len);
    // shard directory (data/shard_*.bin 전부 순서대로)
    DataLoader(const std::string& dir, const std::string& pattern,
               int batch_size, int seq_len);

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_batch();

    size_t total_tokens() const { return total_tok_; }
    size_t tokens_seen()  const { return tokens_seen_; }
    int    epoch()        const { return epoch_; }

private:
    int batch_size_;
    int seq_len_;
    int epoch_ = 0;
    size_t tokens_seen_ = 0;

    // shard 목록
    std::vector<std::string> shards_;
    int cur_shard_ = 0;

    // 현재 청크 (최대 CHUNK_TOKENS 개)
    static constexpr size_t CHUNK_TOKENS = 1 << 22;  // 4M tokens = 8MB
    std::vector<uint16_t> chunk_;
    size_t chunk_pos_ = 0;
    size_t total_tok_ = 0;  // 전체 토큰 수 (추정)

    void load_next_chunk();
    void open_shard(int idx);

    std::ifstream cur_file_;
    size_t shard_tokens_remaining_ = 0;
};