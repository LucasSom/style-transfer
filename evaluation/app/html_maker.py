import os
from typing import List

from utils.files_utils import root_file_name, make_dirs_if_not_exists


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
        * selection_criteria: reason of selection (plagiarism, musicality, etc.)
        * path_orig: path of the original roll
        * path_rec: path of the roll reconstructed (ie, after encode and decode the original matrix)
        * path_transformed: path of the roll after being applied the transformation
    """
    table = f"""
    <h3> A {target} </h3>

      <figure><table>
        <thead>
        <tr><th>Nombre de canción</th><th>Criterio de selección</th><th>Original</th><th>A {target}</th></tr>
        </thead>
        <tbody>
        """
    for s in songs:
        table += "<tr>\n"

        table += f"""    <td>{s['title']}</td>\n"""
        table += f"""    <td>{s['selection_criteria']}</td>\n"""

        for path in s['path_orig'], s['path_transformed']:
            table += f"""    <td><audio controls>
                    <source src="{path}" type="audio/mpeg">
                    Your browser does not support the audio element.
                    </audio></td>\n"""

        table += """</tr>\n"""

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


def make_html(df_transferred, orig, target, app_dir):
    def get_selection_criterion(r):
        try:
            if orig == "Mozart":
                return os.path.basename(root_file_name(r["New audio files"])).split('-')[1].split('_')[1]
            return os.path.basename(root_file_name(r["New audio files"])).split('-')[0].split('_')[1]
        except:
            if orig == "Mozart":
                return os.path.basename(root_file_name(r["New audio files"])).split('-')[2].split('_')[1]
            return os.path.basename(root_file_name(r["New audio files"])).split('-')[1].split('_')[1]

    songs = {target:
            [{'title': os.path.basename(root_file_name(r["New audio files"])).split('-')[0].split('_')[0],
             'selection_criteria': get_selection_criterion(r),
             'path_orig': os.path.join('../audios/', os.path.basename(r["Original audio files"])),
             'path_transformed': os.path.join('../audios/', os.path.basename(r["New audio files"]))
             }
            for i, (_, r) in enumerate(df_transferred.iterrows())
            ]
    }
    file = make_head(orig) + make_body(orig, songs)
    file += """\n<a href="./index.html" class="button">Volver al menú</a>"""

    file_name = f"{app_dir}/audio_{orig}_to_{target}.html"
    make_dirs_if_not_exists(file_name)

    with open(file_name, 'w') as f:
        f.write(file)
        print("Saved HTML file as:", file_name)
