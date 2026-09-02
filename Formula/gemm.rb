class Gemm < Formula
  desc "Local LLM inference server for Apple Silicon (MLX, MCP + REST)"
  homepage "https://github.com/goodmai/basalt"
  url "https://github.com/goodmai/basalt/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_FILLED_AT_RELEASE"
  license "MIT"
  head "https://github.com/goodmai/basalt.git", branch: "main"

  depends_on xcode: ["16.0", :build]
  depends_on arch: :arm64
  depends_on macos: :sequoia

  def install
    system "swift", "build", "-c", "release", "--disable-sandbox"
    # MLX has no metallib of its own in a SwiftPM build: the kernels must be
    # compiled and left next to the binary, or the first inference call dies
    # with "Failed to load the default metallib".
    system "./scripts/build_metal.swift"

    libexec.install ".build/release/Gemm" => "gemm"
    libexec.install ".build/release/mlx.metallib"
    (bin/"gemm").write_env_script libexec/"gemm", {}
  end

  test do
    assert_match "gemm", shell_output("#{bin}/gemm --help")
  end
end
