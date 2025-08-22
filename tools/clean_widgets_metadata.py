#!/usr/bin/env python3
"""
Elimina 'metadata.widgets' de notebooks .ipynb (workaround para el render de GitHub).
Uso:
  python3 tools/clean_widgets_metadata.py            # limpia en .
  python3 tools/clean_widgets_metadata.py path/a/repo
  python3 tools/clean_widgets_metadata.py --dry-run
  python3 tools/clean_widgets_metadata.py --backup
"""
import argparse, json, sys
from pathlib import Path

def clean(root=".", dry_run=False, backup=False):
    changed = 0
    scanned = 0
    for nb_path in Path(root).rglob("*.ipynb"):
        scanned += 1
        try:
            with nb_path.open("r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer {nb_path}: {e}", file=sys.stderr)
            continue

        md = nb.get("metadata", {})
        if "widgets" in md:
            if dry_run:
                print(f"[DRY] Eliminaría metadata.widgets en {nb_path}")
                changed += 1
                continue

            if backup:
                bak = nb_path.with_suffix(nb_path.suffix + ".bak")
                with bak.open("w", encoding="utf-8") as f:
                    json.dump(nb, f, ensure_ascii=False, indent=1)
                    f.write("\n")

            md.pop("widgets", None)
            nb["metadata"] = md

            tmp = nb_path.with_suffix(nb_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
                f.write("\n")
            tmp.replace(nb_path)

            print(f"[OK] Limpio: {nb_path}")
            changed += 1

    print(f"Escaneados: {scanned} | Modificados: {changed}")
    return 0

def main():
    ap = argparse.ArgumentParser(
        description="Eliminar 'metadata.widgets' de notebooks .ipynb."
    )
    ap.add_argument("root", nargs="?", default=".", help="Carpeta raíz (por defecto: .)")
    ap.add_argument("--dry-run", action="store_true", help="No escribe archivos; solo informa.")
    ap.add_argument("--backup", action="store_true", help="Guarda copia .ipynb.bak antes de modificar.")
    args = ap.parse_args()
    sys.exit(clean(args.root, dry_run=args.dry_run, backup=args.backup))

if __name__ == "__main__":
    main()
