import langextract as lx
html_content = lx.visualize("test_output/extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)