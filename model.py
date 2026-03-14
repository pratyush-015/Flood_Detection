import segmentation_models_pytorch as smp


def get_model():

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=8,
        classes=1
    )

    return model