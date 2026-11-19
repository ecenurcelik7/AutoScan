
import cv2
import numpy as np
import os
import glob

IMAGE_DIR = "images"
OUTPUT_DIR = "sonuclar"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------
# 1. NOKTALARI DUZENLE
# -----------------------------------------

def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(d)],
        points[np.argmax(s)],
        points[np.argmax(d)]
    ], dtype=np.float32)


# -----------------------------------------
# 2. PERSPEKTIF DUZELTME
# -----------------------------------------

def rectify_plate(image, box):
    points = order_points(box)

    tl, tr, br, bl = points

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))

    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    if width < 10 or height < 5:
        return None

    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        points, destination
    )

    result = cv2.warpPerspective(
        image, matrix, (width, height)
    )

    if result.shape[0] > result.shape[1]:
        result = cv2.rotate(
            result, cv2.ROTATE_90_CLOCKWISE
        )

    return result


# -----------------------------------------
# 3. ADAYIN PLAKA OLMA PUANI
# -----------------------------------------

def plate_score(plate, ratio, rectangularity):

    if plate is None or plate.size == 0:
        return 0

    h, w = plate.shape[:2]

    if h < 8 or w < 20:
        return 0

    plate = cv2.resize(plate, (320, 80))

    gray = cv2.cvtColor(
        plate, cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        plate, cv2.COLOR_BGR2HSV
    )

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4)
    ).apply(gray)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            binary, 8
        )
    )

    character_count = 0

    for x, y, cw, ch, area in stats[1:]:

        if (
            15 <= ch <= 72
            and 2 <= cw <= 32
            and 15 <= area <= 1200
        ):
            character_count += 1

    # Harf ve rakam sayisi
    if 5 <= character_count <= 11:
        character_score = 1.0
    elif 4 <= character_count <= 13:
        character_score = 0.65
    else:
        character_score = 0.15

    # Plaka zemini ve karakter kontrasti
    contrast_score = np.clip(
        np.std(gray) / 75, 0, 1
    )

    white_ratio = np.mean(gray > 160)
    dark_ratio = np.mean(gray < 90)

    white_score = np.clip(
        1 - abs(white_ratio - 0.55) / 0.55,
        0, 1
    )

    dark_score = np.clip(
        1 - abs(dark_ratio - 0.22) / 0.35,
        0, 1
    )

    # Mavi TR seridi
    blue_mask = cv2.inRange(
        hsv,
        np.array([90, 55, 35]),
        np.array([140, 255, 255])
    )

    left_blue = np.mean(
        blue_mask[:, :45] > 0
    )

    blue_score = np.clip(
        left_blue / 0.25, 0, 1
    )

    # Kenar yogunlugu
    edges = cv2.Canny(gray, 50, 150)

    edge_score = np.clip(
        np.mean(edges > 0) / 0.20,
        0, 1
    )

    # En-boy orani
    ratio_score = np.clip(
        1 - abs(ratio - 4.2) / 3.5,
        0, 1
    )

    score = (
        character_score * 0.28
        + contrast_score * 0.15
        + white_score * 0.12
        + dark_score * 0.10
        + blue_score * 0.12
        + edge_score * 0.10
        + ratio_score * 0.08
        + rectangularity * 0.05
    )

    return float(score)


# -----------------------------------------
# 4. ADAY EKLEME
# -----------------------------------------

def add_candidate(
    candidates, image, box, method
):

    h, w = image.shape[:2]

    box = np.asarray(
        box, dtype=np.float32
    ).reshape(4, 2)

    rect = cv2.minAreaRect(box)

    rw, rh = rect[1]

    if min(rw, rh) < 1:
        return

    long_side = max(rw, rh)
    short_side = min(rw, rh)

    ratio = long_side / short_side

    area_ratio = (
        long_side * short_side
    ) / (w * h)

    if not 2.0 <= ratio <= 7.5:
        return

    if not 0.00025 <= area_ratio <= 0.12:
        return

    if short_side < 10:
        return

    plate = rectify_plate(image, box)

    if plate is None:
        return

    contour_area = cv2.contourArea(
        box.astype(np.float32)
    )

    rectangularity = min(
        contour_area /
        (long_side * short_side + 1e-8),
        1.0
    )

    score = plate_score(
        plate, ratio, rectangularity
    )

    candidates.append({
        "score": score,
        "box": box,
        "plate": plate,
        "method": method
    })


# -----------------------------------------
# 5. BLACKHAT + SOBEL + CANNY
# -----------------------------------------

def generate_edge_candidates(image):

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    candidates = []

    # Blackhat farkli olceklerde denenir.
    for kx, ky in [
        (13, 5),
        (17, 5),
        (21, 7),
        (25, 7),
        (31, 9)
    ]:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kx, ky)
        )

        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT, kernel
        )

        grad = cv2.Sobel(
            blackhat,
            cv2.CV_32F,
            1, 0,
            ksize=3
        )

        grad = cv2.convertScaleAbs(grad)

        grad = cv2.GaussianBlur(
            grad, (5, 5), 0
        )

        _, threshold = cv2.threshold(
            grad, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        processed = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            processed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            add_candidate(
                candidates,
                image,
                box,
                "blackhat_sobel"
            )

    # Canny
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    canny = cv2.Canny(
        blurred, 40, 150
    )

    for kx, ky in [
        (11, 3),
        (17, 5),
        (25, 7),
        (35, 9)
    ]:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kx, ky)
        )

        closed = cv2.morphologyEx(
            canny,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            add_candidate(
                candidates,
                image,
                box,
                "canny"
            )

    # Adaptif esikleme
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9
    )

    for kx, ky in [
        (17, 5),
        (25, 7),
        (35, 9)
    ]:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kx, ky)
        )

        closed = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            add_candidate(
                candidates,
                image,
                box,
                "adaptive"
            )

    return candidates


# -----------------------------------------
# 6. MAVI TR SERIDI ILE ADAY URETME
# -----------------------------------------

def generate_blue_candidates(image):

    hsv = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        np.array([90, 55, 35]),
        np.array([140, 255, 255])
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 9)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        center = np.array(
            rect[0], dtype=np.float32
        )

        rw, rh = rect[1]

        if min(rw, rh) < 5:
            continue

        # Mavi seridin uzun ekseni plaka yuksekligidir.
        if rw > rh:
            height = rw
            short_width = rh
            angle = np.deg2rad(rect[2])
            long_axis = np.array([
                np.cos(angle),
                np.sin(angle)
            ])
        else:
            height = rh
            short_width = rw
            angle = np.deg2rad(rect[2] + 90)
            long_axis = np.array([
                np.cos(angle),
                np.sin(angle)
            ])

        if height < 12 or height > image.shape[0] * 0.16:
            continue

        if short_width > height * 0.8:
            continue

        # Plakanin yatay ekseni, mavi seridin
        # uzun eksenine diktir.
        direction = np.array([
            long_axis[1],
            -long_axis[0]
        ])

        # Plakanin devam ettigi yonu sec.
        if direction[0] < 0:
            direction = -direction

        vertical = long_axis * (height / 2)

        # Farkli plaka genislikleri denenir.
        for factor in [
            3.7, 4.0, 4.3, 4.6, 4.9, 5.2
        ]:

            total_width = height * factor

            # Mavi seridin sag kenarindan basla.
            left_center = (
                center
                + direction * (short_width / 2)
            )

            right_center = (
                left_center
                + direction * total_width
            )

            box = np.float32([
                left_center - vertical,
                right_center - vertical,
                right_center + vertical,
                left_center + vertical
            ])

            add_candidate(
                candidates,
                image,
                box,
                "blue_band"
            )

    return candidates


# -----------------------------------------
# 7. ADAYLARI BIRLESTIR VE TEK SONUC SEC
# -----------------------------------------

def overlap(box_a, box_b):

    a = cv2.boundingRect(
        box_a.astype(np.int32)
    )

    b = cv2.boundingRect(
        box_b.astype(np.int32)
    )

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    union = aw * ah + bw * bh - inter

    return inter / (union + 1e-8)


def detect_best_plate(image):

    candidates = []

    # Butun yontemleri calistir.
    candidates.extend(
        generate_edge_candidates(image)
    )

    candidates.extend(
        generate_blue_candidates(image)
    )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Ayni bolgeyi tekrar tekrar secme.
    # Geriye kalanlar arasindan tek en iyi adayi al.
    best = None

    for candidate in candidates:

        if best is None:
            best = candidate
            continue

        if overlap(
            candidate["box"],
            best["box"]
        ) > 0.45:

            if candidate["score"] > best["score"]:
                best = candidate

    return best


# -----------------------------------------
# 8. FOTOGRAFI ISLE
# -----------------------------------------

def process_image(image_path):

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    print("\nIsleniyor:", filename)

    image = cv2.imread(image_path)

    if image is None:
        print("Fotograf okunamadi.")
        return

    best = detect_best_plate(image)

    if best is None:

        print("Plaka bulunamadi.")

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_bulunamadi.png"
            ),
            image
        )

        return

    result = image.copy()

    box = best["box"].astype(np.int32)

    # Tek plaka cercevesi.
    cv2.polylines(
        result,
        [box],
        True,
        (0, 255, 0),
        3
    )

    # Puan bilgisi.
    score_text = (
        f"Plaka: %{best['score'] * 100:.1f}"
    )

    x, y = box[0]

    cv2.putText(
        result,
        score_text,
        (int(x), max(25, int(y) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Tek tespit gorseli.
    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_tespit.png"
        ),
        result
    )

    # Tek plaka kirpimi.
    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_plaka.png"
        ),
        best["plate"]
    )

    print("Tek plaka tespiti kaydedildi.")
    print("Yontem:", best["method"])
    print("Puan:", round(best["score"], 3))


# -----------------------------------------
# 9. ANA PROGRAM
# -----------------------------------------

def main():

    image_files = []

    for extension in (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp"
    ):

        image_files.extend(
            glob.glob(
                os.path.join(
                    IMAGE_DIR,
                    extension
                )
            )
        )

    image_files.sort()

    if not image_files:

        print(
            "images klasorunde fotograf bulunamadi."
        )

        return

    print(
        "Toplam fotograf:",
        len(image_files)
    )

    for image_path in image_files:
        process_image(image_path)

    print("\n------------------------------")
    print("AUTO SCAN TAMAMLANDI")
    print("Her fotograf icin tek sonuc kaydedildi.")
    print("Sonuclar klasorunu kontrol et.")
    print("------------------------------")


if __name__ == "__main__":
    main()
