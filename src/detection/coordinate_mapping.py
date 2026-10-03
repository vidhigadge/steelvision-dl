def map_box_to_original(box, processed_size, original_size):
    """
    Map an XYXY bounding box from a processed image
    back to the original image coordinates.

    box: (x1, y1, x2, y2)
    processed_size: (width, height)
    original_size: (width, height)
    """
    x1, y1, x2, y2 = box
    processed_w, processed_h = processed_size
    original_w, original_h = original_size

    scale_x = original_w / processed_w
    scale_y = original_h / processed_h

    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    )