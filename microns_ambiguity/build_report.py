"""README -> self-contained HTML report (figures inlined) at outputs/report.html."""
import base64
import re

import markdown

from .config import OUT, ROOT

HEAD = open(ROOT / "report_head.html").read()


def main():
    md = (ROOT / "README.md").read_text()
    md = re.sub(r"\n## Run\n.*?(?=\n## Results)", "\n", md, flags=re.S)
    md = md.replace("# Representational ambiguity in mouse visual cortex (MICrONS)\n", "")
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])

    def inline(m):
        data = base64.b64encode((ROOT / m.group(2)).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{m.group(1)}"></figure>'

    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" /></p>', inline, html)
    html = html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    (OUT / "report.html").write_text(HEAD + '<article>' + html + "</article></div>")
    print("report written")


if __name__ == "__main__":
    main()
