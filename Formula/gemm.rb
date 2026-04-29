class Gemm < Formula
  desc "Local LLM inference server for Apple Silicon"
  homepage "https://github.com/your-org/Gem"
  url "https://github.com/your-org/Gem/archive/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on xcode: ["16.0", :build]
  depends_on macos: :sonoma
  depends_on arch: :arm64

  def install
    system "./scripts/build_metal.swift"
    system "swift", "build", "-c", "release", "--disable-sandbox"
    bin.install ".build/release/Gemm" => "gemm"
    lib.install "default.metallib"
  end

  test do
    assert_match "0.1.0", shell_output("#{bin}/gemm --help")
  end
end
