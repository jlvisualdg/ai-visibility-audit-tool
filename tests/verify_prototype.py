"""
Ad-hoc verification script for the aeo-audit-tool.

Per the aeo-tracking skill's verification policy: this script lives at the
system-required temp path, uses tempfile.mkdtemp for isolated state,
exercises the full pipeline end-to-end, and is deleted after the run.

The summary line is intentionally "ad-hoc verification" not "suite green"
— that's the policy-required wording.
"""

import os
import sys
import tempfile
import shutil
import re
import json
from pathlib import Path


def main():
    project_root = Path(os.environ.get("PROJECT_ROOT", "/Users/mac/Desktop/aeo-audit-tool"))
    sys.path.insert(0, str(project_root))

    # Isolated state dir so the verification doesn't pollute output/
    tmp = Path(tempfile.mkdtemp(prefix="aeo-audit-verify-"))
    print(f"Using isolated output dir: {tmp}")

    try:
        from src.crawler import crawl_domain, generate_buyer_topics
        from src.visibility import build_citation_matrix, build_engines
        from src.analyzer import analyze
        from src.reporter import generate_report

        # 1. Crawl a real (safe) site — example.com is a known-stable test domain
        print("\n[1/5] Crawl")
        crawl = crawl_domain("example.com", max_pages=3, timeout=10)
        assert crawl.pages_crawled >= 1, "no pages crawled"
        assert crawl.health_score > 0, "no health score"
        assert len(crawl.issues) >= 1, "no issues recorded"
        print(f"  pages={crawl.pages_crawled} health={crawl.health_score} issues={len(crawl.issues)}")

        # 2. Topics
        print("\n[2/5] Buyer topics")
        topics = generate_buyer_topics(crawl)
        assert 1 <= len(topics) <= 4, f"unexpected topic count {len(topics)}"
        for t in topics:
            assert len(t) >= 3, f"topic too short: {t!r}"
        print(f"  {topics}")

        # 3. Citation matrix (mock mode — no API key needed)
        print("\n[3/5] Citation matrix (mock mode)")
        engines = build_engines("example.com")
        assert all(not e.is_real() for e in engines), "expected all mocks when no key"
        for e in engines:
            assert "(mock)" in e.name, f"mock engine {e.name!r} missing (mock) suffix"
        matrix = build_citation_matrix("example.com", topics, passes=1, engines=engines)
        assert matrix.total_cells == len(topics) * len(engines)
        assert len(matrix.results) == matrix.total_cells
        for r in matrix.results:
            assert r.engine in [e.name for e in engines]
            assert r.pass_count == 1
        print(f"  cells={matrix.total_cells} all-mock={all(not e.is_real() for e in engines)}")

        # 4. Analyze
        print("\n[4/5] Strategic analysis")
        report = analyze(crawl, matrix, topics=topics)
        assert report.domain == "example.com"
        assert 0.0 <= report.ai_coverage_pct <= 100.0
        assert len(report.fixes) >= 1
        assert report.strategic_read.title
        assert report.strategic_read.summary
        assert report.strategic_read.the_play
        print(f"  coverage={report.ai_coverage_pct}% fixes={len(report.fixes)} title={report.strategic_read.title!r}")

        # 5. Report
        print("\n[5/5] HTML report")
        out_path = generate_report(report, output_dir=str(tmp), no_ai=False)
        assert Path(out_path).exists()
        html = Path(out_path).read_text()
        assert "example.com" in html
        assert "Citation matrix" in html
        assert "Crawl signals" in html
        assert "Prioritized fixes" in html
        # Mock state must be visible
        assert "(mock)" in html, "mock state not surfaced in report"
        # Dark theme markers
        assert "#0a0e27" in html
        assert "#00d4ff" in html
        # Lead-gen CTA
        assert "Book a workshop" in html
        size_kb = len(html) / 1024
        print(f"  path={out_path}")
        print(f"  size={size_kb:.1f}KB")
        print(f"  has_all_sections=yes")
        print(f"  mock_visible=yes")

        print("\nAD-HOC VERIFICATION: all phases passed end-to-end.")
        return 0

    except AssertionError as e:
        print(f"\nAD-HOC VERIFICATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nAD-HOC VERIFICATION FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
