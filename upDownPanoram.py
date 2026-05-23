# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuración ──────────────────────────────────────────────────────────────
image_paths = [
    "1de11fa6-f58a-4c19-988f-dd5ddc3467f9.jpg",   # img 1 (inferior)
    "ced57e85-47d1-482e-88eb-a9860d15da24.jpg",   # img 2 (inferior)
    "0279053f-c683-4fb7-85f8-32ca828f9dd6.jpg",   # img 3 (inferior)
    "5f12e1d5-c055-4950-b388-cd282c2c474d.jpg",   # img 4 (superior)
    "1cb5998f-c8e0-47f7-878c-1fa38f4d5fc5.jpg",   # img 5 (superior)
    "c9dd7eb3-3309-4ee9-b6b8-21d026cb415a.jpg",   # img 6 (superior)
]

OUTPUT_DIR = r"C:\Users\Usuario\OneDrive\Documentos\Programación\Vision_Dron\panoramas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Funciones base
# ─────────────────────────────────────────────────────────────────────────────

def detect_features(image_bgr, n_features=3000):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio * n.distance]
    return good


def compute_homography(kp_src, kp_dst, matches, reproj_thresh=4.0):
    if len(matches) < 4:
        raise ValueError(f"Se necesitan al menos 4 matches, se tienen {len(matches)}")
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"    Inliers RANSAC: {inliers}/{len(matches)}")
    return H


def warp_and_blend(img_src, img_dst, H):
    h_src, w_src = img_src.shape[:2]
    h_dst, w_dst = img_dst.shape[:2]

    corners_src = np.float32(
        [[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]
    ).reshape(-1, 1, 2)
    corners_proj = cv2.perspectiveTransform(corners_src, H)

    all_corners = np.concatenate([
        corners_proj,
        np.float32([[0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]]).reshape(-1, 1, 2)
    ], axis=0)

    x_min, y_min = np.floor(all_corners[:, 0, :].min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners[:, 0, :].max(axis=0)).astype(int)

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0,      1]], dtype=np.float64)

    warped = cv2.warpPerspective(img_src, T @ H, (x_max - x_min, y_max - y_min))

    roi = warped[-y_min:-y_min + h_dst, -x_min:-x_min + w_dst]
    mask_dst = np.all(roi == 0, axis=2)
    roi[~mask_dst] = img_dst[~mask_dst]
    roi[mask_dst]  = img_dst[mask_dst]

    return warped


def stitch_pair(img_a, img_b, label, show=True):
    """
    Une img_a con img_b.
    Devuelve (resultado_BGR, matches_encontrados) para poder
    evaluar compatibilidad aunque la fusion falle.
    """
    print(f"\n{'='*50}")
    print(f"  Probando: {label}")
    print(f"{'='*50}")

    kp_a, desc_a = detect_features(img_a)
    kp_b, desc_b = detect_features(img_b)
    matches = match_features(desc_a, desc_b)
    print(f"  Matches buenos: {len(matches)}", end="")

    # Diagnostico rapido de compatibilidad
    if len(matches) >= 100:
        print("  -> EXCELENTE traslape")
    elif len(matches) >= 30:
        print("  -> traslape ACEPTABLE")
    elif len(matches) >= 4:
        print("  -> traslape DEBIL (resultado puede ser incorrecto)")
    else:
        print("  -> SIN traslape suficiente, no se puede fusionar")
        return None, len(matches)

    try:
        H = compute_homography(kp_a, kp_b, matches)
        result = warp_and_blend(img_a, img_b, H)

        out_path = os.path.join(OUTPUT_DIR, label + ".jpg")
        cv2.imwrite(out_path, result)
        print(f"  Guardado: {out_path}")

        if show:
            plt.figure(figsize=(18, 5))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f"{label}  |  matches: {len(matches)}", fontsize=13)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return result, len(matches)

    except Exception as e:
        print(f"  ERROR al fusionar: {e}")
        return None, len(matches)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de imagenes
# ─────────────────────────────────────────────────────────────────────────────

print("Cargando imagenes...")
images = []
for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"No se encontro: {p}")
    images.append(img)
    print(f"  OK {os.path.basename(p)}  ->  {img.shape[1]}x{img.shape[0]} px")

img1, img2, img3, img4, img5, img6 = images

# ─────────────────────────────────────────────────────────────────────────────
# PRUEBA DE COMPATIBILIDAD — una imagen inferior vs su correspondiente superior
#   P14: img1 (inf) + img4 (sup)
#   P25: img2 (inf) + img5 (sup)
#   P36: img3 (inf) + img6 (sup)
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n=== PRUEBA DE COMPATIBILIDAD inf/sup ===")

resultados = {}

P14, m14 = stitch_pair(img1, img4, "P14")
P25, m25 = stitch_pair(img2, img5, "P25")
P36, m36 = stitch_pair(img3, img6, "P36")

# ─────────────────────────────────────────────────────────────────────────────
# Resumen de compatibilidad
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'*'*50}")
print("  RESUMEN DE COMPATIBILIDAD:")
print(f"    P14  (img1 + img4):  {m14} matches  {'OK' if m14 >= 30 else 'FALLO' if m14 < 4 else 'DEBIL'}")
print(f"    P25  (img2 + img5):  {m25} matches  {'OK' if m25 >= 30 else 'FALLO' if m25 < 4 else 'DEBIL'}")
print(f"    P36  (img3 + img6):  {m36} matches  {'OK' if m36 >= 30 else 'FALLO' if m36 < 4 else 'DEBIL'}")
print(f"{'*'*50}\n")