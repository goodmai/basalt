class Basalt < Formula
  desc "Local LLM inference server for Apple Silicon (MLX, MCP + REST)"
  homepage "https://github.com/goodmai/basalt"
  url "https://github.com/goodmai/basalt/archive/refs/tags/v0.9.0.tar.gz"
  sha256 "f2b314c4e8f3c7d33a9c699667bf2eb2f046af9a52f03a395037f850a19024ec"
  license "MIT"
  head "https://github.com/goodmai/basalt.git", branch: "main"

  depends_on xcode: ["27.0", :build]
  depends_on arch: :arm64
  depends_on macos: :golden_gate

  def install
    unless quiet_system("xcrun", "metal", "--version")
      odie <<~EOS
        The Metal toolchain is not installed, so MLX's kernels cannot be built.
        Install it once (a few GB, from Apple), then retry:

          xcodebuild -downloadComponent MetalToolchain
      EOS
    end

    system "swift", "build", "-c", "release", "--disable-sandbox"
    system "./scripts/build_metal.swift"

    bin_path = Utils.safe_popen_read("swift", "build", "-c", "release", "--disable-sandbox", "--show-bin-path").chomp
    libexec.install "#{bin_path}/Basalt" => "basalt"
    libexec.install "#{bin_path}/mlx.metallib"
    (bin/"basalt").write_env_script libexec/"basalt", {}
  end

  test do
    assert_match "basalt", shell_output("#{bin}/basalt --help")
  end
end
