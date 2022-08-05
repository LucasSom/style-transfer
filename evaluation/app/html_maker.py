import os
from typing import List

from utils.files_utils import data_path, project_path, root_file_name


def make_head(title: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
    
        <link rel="stylesheet" media="screen" href="style.css?v=8may2013">
    </head>
    """


def make_table(target: str, songs: List[dict]) -> str:
    """
    :param target: name of the style to which the subdataset was converted.
    :param songs: list of dictionary with keys:
        * title: title of the song
        * path_orig: path of the original roll
        * path_rec: path of the roll reconstructed (ie, after encode and decode the original matrix)
        * path_transformed: path of the roll after being applied the transformation
    """
    table = f"""
    <h3> A {target} </h3>

      <figure><table>
        <thead>
        <tr><th>Nombre de canción</th><th>Original</th><th>Reconstrucción</th><th>A {target}</th><th>Opinión</th></tr>
        </thead>
        <tbody>
        """
    for s in songs:
        table += "<tr>\n"

        table += f"""    <td>{s['title']}</td>"""

        for path in s['path_orig'], s['path_recon'], s['path_transformed']:
            table += f"""    <td><audio controls>
                    <source src="{path}" type="audio/mpeg">
                    Your browser does not support the audio element.
                    </audio></td>"""

        table += """<td><input type="text"></td>
            </tr>\n"""

    return table + """  
        </tbody>
        </table></figure>
    
      <br/>
    """


def make_body(original_style: str, songs: dict) -> str:
    file = f"""<body id="css-zen-garden">
    <div class="page-wrapper">
    
    <h1>Tabla de audios coder-decoder</h1>
    
    <h2>{original_style}</h2>
    """

    for target, transformed_songs in songs.items():
        file += make_table(target, transformed_songs)

    file += "</div>\n"
    return file + "</body>"


def make_html(df_transferred, orig, targets, audios_path=f"{data_path}Audios/"):
    songs = {
        t: [{'title': root_file_name(r.Title),
             'path_orig': f"{audios_path}/{root_file_name(r.Title)}_orig_{i+1}.mp3",
             'path_recon': f"{audios_path}/{root_file_name(r.Title)}_recon_{i+1}.mp3",
             'path_transformed': f"{audios_path}/{root_file_name(r.Title)}_{orig}_to_{t}_{i+1}.mp3"
             }
            for i, (_, r) in enumerate(df_transferred.iterrows())
            ]

        for t in targets
    }
    file = make_head(orig) + make_body(orig, songs)
    file += """\n<a href="./index.html" class="button">Volver al menú</a>"""

    file_name = os.path.join(project_path, f"evaluation/app/audio_{orig}.html")
    with open(file_name, 'w') as f:
        f.write(file)
        # if verbose: print("Saved as:", file_name)
