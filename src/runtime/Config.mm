#include "Config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

void Config::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "[Config] Cannot open: " << filepath << "\n";
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Strip comments
        auto sharp = line.find('#');
        if (sharp != std::string::npos) line.resize(sharp);
        // Strip whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        auto end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line.resize(end + 1);
        // Parse KEY=VALUE
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // Trim key and value
        key.erase(key.find_last_not_of(" \t") + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        if (!key.empty())
            settings[key] = val;
    }
}

std::string Config::getString(const std::string& key, const std::string& def) {
    auto it = settings.find(key);
    return (it != settings.end()) ? it->second : def;
}

int Config::getInt(const std::string& key, int def) {
    auto it = settings.find(key);
    if (it == settings.end()) return def;
    try { return std::stoi(it->second); }
    catch (...) { return def; }
}

float Config::getFloat(const std::string& key, float def) {
    auto it = settings.find(key);
    if (it == settings.end()) return def;
    try { return std::stof(it->second); }
    catch (...) { return def; }
}

bool Config::getBool(const std::string& key, bool def) {
    auto it = settings.find(key);
    if (it == settings.end()) return def;
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "1" || v == "true" || v == "yes");
}

void Config::print() const {
    std::cout << "═══ Config ═══\n";
    for (auto& [k, v] : settings)
        std::cout << "  " << k << " = " << v << "\n";
    std::cout << "══════════════\n\n";
}
