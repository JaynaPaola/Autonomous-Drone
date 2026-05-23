# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Forzar UTF-8 en la salida del terminal (necesario en Windows)
sys.stdout.reconfigure(encoding="utf-8")

# ── Configuración: pon aquí tus imágenes EN EL ORDEN que quieras ──────────────
image_paths = [
    "1.jpeg",   # img 1
    "2.jpeg",   # img 2
    "3.jpeg",   # img 3
    "4.jpeg",   # img 4
    "5.jpeg",   # img 5
    "6.jpeg",   # img 6
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

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    warped = cv2.warpPerspective(img_src, T @ H, (canvas_w, canvas_h))

    roi = warped[-y_min:-y_min + h_dst, -x_min:-x_min + w_dst]
    mask_dst = np.all(roi == 0, axis=2)
    roi[~mask_dst] = img_dst[~mask_dst]
    roi[mask_dst]  = img_dst[mask_dst]

    return warped


def stitch_pair(img_a, img_b, label, show=True):
    """
    Une img_a con img_b, guarda el resultado y lo muestra.
    Devuelve la imagen fusionada en BGR.
    """
    print(f"\n{'='*50}")
    print(f"  Fusionando: {label}")
    print(f"{'='*50}")

    kp_a, desc_a = detect_features(img_a)
    kp_b, desc_b = detect_features(img_b)
    matches = match_features(desc_a, desc_b)
    print(f"  Matches buenos: {len(matches)}")

    H = compute_homography(kp_a, kp_b, matches)
    result = warp_and_blend(img_a, img_b, H)

    out_path = os.path.join(OUTPUT_DIR, label + ".jpg")
    cv2.imwrite(out_path, result)
    print(f"  Guardado: {out_path}")

    if show:
        plt.figure(figsize=(18, 5))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Panorama: {label}", fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Carga de imagenes
# ─────────────────────────────────────────────────────────────────────────────

print("Cargando imagenes...")
images = []
for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"No se encontro o no se pudo leer: {p}")
    images.append(img)
    print(f"  OK {os.path.basename(p)}  ->  {img.shape[1]}x{img.shape[0]} px")

img1, img2, img3, img4, img5, img6 = images

# ─────────────────────────────────────────────────────────────────────────────
# GRUPO A — parte inferior de la pista (imgs 1, 2, 3)
#   Paso A1: 1 + 2   -> P12
#   Paso A2: P12 + 3 -> P123
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n=== GRUPO A: imagenes 1-2-3 (parte inferior) ===")

P12  = stitch_pair(img1, img2, "P12")
P123 = stitch_pair(P12,  img3, "P123")

# ─────────────────────────────────────────────────────────────────────────────
# GRUPO B — parte superior de la pista (imgs 4, 5, 6)
#   Paso B1: 4 + 5   -> P45
#   Paso B2: P45 + 6 -> P456
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n=== GRUPO B: imagenes 4-5-6 (parte superior) ===")

P45  = stitch_pair(img4, img5, "P45")
P456 = stitch_pair(P45,  img6, "P456")

# ─────────────────────────────────────────────────────────────────────────────
# FUSION FINAL — P456 (superior) + P123 (inferior)
#
# IMPORTANTE: esta fusion requiere traslape visual entre ambos panoramas.
# Si las imagenes no comparten zona en comun, ORB no encontrara matches
# y veras el error "Se necesitan al menos 4 matches".
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n=== FUSION FINAL: P456 (superior) + P123 (inferior) ===")

PANORAMA_FINAL = stitch_pair(P456, P123, "PANORAMA_FINAL")

# ─────────────────────────────────────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'*'*50}")
print("  Todos los resultados en:", OUTPUT_DIR)
print("  Archivos: P12.jpg | P123.jpg | P45.jpg | P456.jpg | PANORAMA_FINAL.jpg")
print(f"{'*'*50}\n")