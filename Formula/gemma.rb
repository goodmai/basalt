class Gemma < Formula
  desc "Local LLM inference server for Apple Silicon"
  homepage "https://github.com/your-org/GemmaServer"
  url "https://github.com/your-org/GemmaServer/archive/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on xcode: ["16.0", :build]
  depends_on macos: :sequoia
  depends_on arch: :arm64

  def install
    system "swift", "build", "-c", "release", "--disable-sandbox"
    bin.install ".build/release/GemmaServer" => "gemma"
  end

  test do
    assert_match "0.1.0", shell_output("#{bin}/gemma --help")
  end
end
