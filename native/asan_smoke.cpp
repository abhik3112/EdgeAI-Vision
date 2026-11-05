#include <array>
#include <iostream>
#include <string_view>

int main() {
    std::array<std::string_view, 3> components{
        "image", "model", "edge-runtime"
    };

    for (std::size_t i = 0; i < components.size(); ++i) {
        std::cout << components[i] << '\n';
    }

    std::cout << "ASan smoke test completed without memory errors.\n";
    return 0;
}
