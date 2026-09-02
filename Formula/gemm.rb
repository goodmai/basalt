class Gemm < Formula
  desc "Local LLM inference server for Apple Silicon (MLX, MCP + REST)"
  homepage "https://github.com/goodmai/basalt"
  url "https://github.com/goodmai/basalt/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "512d7e9cf54c92402416a8b8331d5a15a984bdf9701c11299f52235b3c5ae580"
  license "MIT"
  head "https://github.com/goodmai/basalt.git", branch: "main"

  depends_on xcode: ["16.0", :build]
  depends_on arch: :arm64
  depends_on macos: :sequoia

  def install
    # Xcode 26 moved the Metal compiler into a separately downloaded component,
    # and there is no way around needing it: a SwiftPM build of mlx-swift ships
    # no metallib, so the kernels have to be compiled here.
    unless quiet_system("xcrun", "metal", "--version")
      odie <<~EOS
        The Metal toolchain is not installed, so MLX's kernels cannot be built.
        Install it once (a few GB, from Apple), then retry:

          xcodebuild -downloadComponent MetalToolchain
      EOS
    end

    system "swift", "build", "-c", "release", "--disable-sandbox"
    # Leaves mlx.metallib next to the binary — the first path MLX searches.
    # Without it the first inference call dies with
    # "Failed to load the default metallib".
    system "./scripts/build_metal.swift"

    libexec.install ".build/release/Gemm" => "gemm"
    libexec.install ".build/release/mlx.metallib"
    (bin/"gemm").write_env_script libexec/"gemm", {}
  end

  test do
    assert_match "gemm", shell_output("#{bin}/gemm --help")
  end
end
