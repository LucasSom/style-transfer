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
        * audio_path_orig: path of the audio of the original roll
        * audio_path_rec: path of audio of the reconstructed roll
        * audio_path_transformed: path of the audio after being applied the transformation
        * sheet_path_orig: path of the sheet of the original roll
        * sheet_path_rec: path of sheet of the reconstructed roll
        * sheet_path_transformed: path of the sheet after being applied the transformation
    """
    table = f"""
    <h3> A {target} </h3>

      <figure><table>
        <thead>
        <tr>
        <th>Nombre de canción</th><th>Criterio de selección</th>
        <th>Audio original</th><th>Audio reconstruido</th><th>Audio transformado</th>
        <th>Partitura original</th><th>Partitura reconstruida</th><th>Partitura transformada</th>
        </tr>
        </thead>
        <tbody>
        """
    for s in songs:
        table += "<tr>\n"

        table += f"""    <td>{s['title']}</td>\n"""
        table += f"""    <td>{s['selection_criteria']}</td>\n"""

        relative_path = '../../../../preprocessed_data/original/audios/4bars/' + s['audio_path_orig'].split('/')[-1]
        table += f"""    <td><audio controls>
                            <source src="{relative_path}" type="audio/mpeg">
                            Your browser does not support the audio element.
                            </audio></td>\n"""
        for path in s['audio_path_rec'], s['audio_path_transformed']:
            relative_path = '../../audios/' + path.split('/')[-1]
            table += f"""    <td><audio controls>
                    <source src="{relative_path}" type="audio/mpeg">
                    Your browser does not support the audio element.
                    </audio></td>\n"""

        for path in s['sheet_path_orig'], s['sheet_path_rec'], s['sheet_path_transformed']:
            relative_path = '../../sheets/' + path.split('/')[-1]
            table += f"""    <td><img src="{relative_path}"></td>\n"""

        table += """</tr>\n"""

    return table + """  
        </tbody>
        </table></figure>
    
      <br/>
    """


def make_body(original_style: str, mutation: str, songs: dict) -> str:
    file = f"""<body id="css-zen-garden">
    <div class="page-wrapper">
    
    <h1>Tabla de audios transformados con: {mutation}</h1>
    
    <h2>{original_style}</h2>
    """

    for target, transformed_songs in songs.items():
        file += make_table(target, transformed_songs)

    file += "</div>\n"
    return file + "</body>"


def make_html(df, orig, target, app_dir, mutation):
    songs = {target: [{'title': r['Title'],
                       'selection_criteria': r['Selection criteria'],
                       'audio_path_orig': r["Original audio files"],
                       'audio_path_rec': r["Reconstructed audios"],
                       'audio_path_transformed': r["New audio files"],
                       'sheet_path_orig': r["Original sheet"],
                       'sheet_path_rec': r["Reconstructed sheet"],
                       'sheet_path_transformed': r["New sheet"]
                       }
                      for _, r in df.iterrows()
                      ]
             }
    file = make_head(orig) + make_body(orig, mutation, songs)
    file += f"""\n<a href="./index-{mutation}.html" class="button">Volver al menú</a>"""

    file_name = f"{app_dir}{orig}_to_{target}-{mutation}.html"
    make_dirs_if_not_exists(os.path.dirname(file_name))

    with open(file_name, 'w') as f:
        f.write(file)
        print("Saved HTML file as:", file_name)


def make_index(mutation, app_path, files):
    file = make_head("Evaluación")

    file += f"""<body id="css-zen-garden">
    <div class="page-wrapper">
    
    <h1>Transformaciones disponibles para {mutation}</h1>
    
    <ul>
    """

    for transformation in files:
        file += f"""<li><a href="./{transformation}-{mutation}.html" class="button">{transformation}</a></li>\n"""

    file += "</ul>\n</div>\n"
    file += "</body>"

    file_name = f"{app_path}index-{mutation}.html"
    make_dirs_if_not_exists(os.path.dirname(file_name))

    with open(file_name, 'w') as f:
        f.write(file)
        print("Saved HTML file as:", file_name)
