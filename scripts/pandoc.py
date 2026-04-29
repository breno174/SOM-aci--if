from markdown import markdown
from htmldocx import HtmlToDocx
from docx import Document


def main():
    # 1. Ler o arquivo Markdown
    with open("docs/relatorio_metodologia_som.md", "r", encoding="utf-8") as f:
        texto_md = f.read()

    # 2. Converter Markdown para HTML
    html = markdown(texto_md)

    # 3. Criar um documento Word e converter o HTML para ele
    documento = Document()
    parser = HtmlToDocx()
    parser.add_html_to_document(html, documento)

    # 4. Salvar
    documento.save("docs/relatorio_metodologia_som.docx")
    print("Arquivo convertido com sucesso!")


if __name__ == "__main__":
    main()
