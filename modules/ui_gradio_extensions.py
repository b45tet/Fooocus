# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import re
import args_manager

from modules.localization import localization_js


modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'gradio_api/file={web_path}?{os.path.getmtime(fn)}'


JAVASCRIPT_FILES = ['script.js', 'contextMenus.js', 'localization.js', 'edit-attention.js', 'viewer.js', 'imageviewer.js']


def javascript_html():
    # Gradio 5 appends external head scripts asynchronously (out of order, after DOMContentLoaded),
    # so the scripts are inlined into a single block that runs in order.
    sources = [localization_js(args_manager.args.language)]
    for name in JAVASCRIPT_FILES:
        with open(os.path.join(script_path, 'javascript', name), encoding='utf-8') as f:
            sources.append(f.read())

    if args_manager.args.theme:
        sources.append(f'set_theme("{args_manager.args.theme}");')

    code = re.sub(r'</(script)', r'<\\/\1', ';\n'.join(sources), flags=re.IGNORECASE)
    samples_path = webpath('sdxl_styles/samples/fooocus_v2.jpg')
    head = f'<meta name="samples-path" content="{samples_path}">\n'
    head += f'<script type="text/javascript">{code}</script>\n'
    return head
