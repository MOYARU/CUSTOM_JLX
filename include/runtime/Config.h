#pragma once

#include <string>
#include <map>

class Config {
public:
    static Config& getInstance();

    void load(const std::string& filepath);

    // Getters with default values
    std::string getString(const std::string& key, const std::string& def = "");
    int getInt(const std::string& key, int def = 0);
    float getFloat(const std::string& key, float def = 0.0f);
    bool getBool(const std::string& key, bool def = false);

    void print() const;

private:
    Config() = default;
    std::map<std::string, std::string> settings;
};
