import cv2
import numpy as np

def colorize_image(input_image_path, output_image_path):
    # Model paths
    proto_file = "./models/colorization_deploy_v2.prototxt"
    model_file = "./models/colorization_release_v2.caffemodel"
    kernel_file = "./models/pts_in_hull.npy"

    # Load the model
    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

    # Load cluster centers
    pts = np.load(kernel_file)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

    # Read the input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not read input image at {input_image_path}")
        return
    h, w = img.shape[:2]

    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_l = img_lab[:, :, 0]

    # Resize the L channel to 224x224 (model's input size)
    img_l_resized = cv2.resize(img_l, (224, 224))
    img_l_resized = img_l_resized.astype("float32") / 255.0
    img_l_resized = img_l_resized - 0.5  # Normalize to [-0.5, 0.5]
    img_l_resized = img_l_resized[np.newaxis, np.newaxis, :, :]

    # Perform forward pass
    net.setInput(cv2.dnn.blobFromImage(img_l_resized))
    ab_decoded = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the output to the original image size
    ab_decoded_resized = cv2.resize(ab_decoded, (w, h))

    # Combine with the original L channel
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_decoded_resized), axis=2)
    img_bgr_out = cv2.cvtColor(img_lab_out, cv2.COLOR_LAB2BGR)
    img_bgr_out = np.clip(img_bgr_out, 0, 255).astype("uint8")

    # Save the colorized image
    cv2.imwrite(output_image_path, img_bgr_out)
    print(f"Colorized image saved to {output_image_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "./models/mountain.jpg"  # Path to grayscale input image
    output_image_path = "./models/colorized_image.jpg"  # Path to save the colorized image
    colorize_image(input_image_path, output_image_path)
