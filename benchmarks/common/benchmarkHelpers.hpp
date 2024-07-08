#include <iostream>
#include <stdexcept>
#include <array>
#include <memory>
#include <string>
#include <sstream>

std::string exec(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main() {
    try {
        // Execute the git command to get the current commit hash
        std::string commitHash = exec("git rev-parse HEAD");
        // Remove the trailing newline character, if any
        if (!commitHash.empty() && commitHash.back() == '\n') {
            commitHash.pop_back();
        }
        std::cout << "Current commit hash: " << commitHash << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

  try {
        std::string repoUrl = exec("git config --get remote.origin.url");
        if (!repoUrl.empty()) {
            repoUrl.pop_back(); // Remove trailing newline
        }
        std::cout << "Repository URL: " << repoUrl << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }



    return 0;
}



