# Классификация известных личностей по изображению лица
**Задача проекта** - **классификация пяти знаменитых личностей** (Билл Гейтс, Илон Маск, Джефф Безос, Марк Цукерберг, Стив Джобс) **по изображению лица** с использованием **глубокого обучения** (*CNN*).

Ноутбук со всеми этапами проекта находится [<b>тут</b>](./Celebrity%20classification.ipynb)

Используемый фреймворк для *DL* - [<b><i>PyTorch</i></b>](https://pytorch.org/)

Модель, взятая за основу для *fine-tuning* - [<b><i>ResNeXt</i> с 50-ю слоями</b>](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html)

Результаты:
* $\text{Accuracy}_{\text{test}} \approx 0.97$
* $F_1\text{-score}_{\text{test}} \approx 0.97$

## Использование готовой модели в *Python*
1. *Python 11.x*

2. Установить [<b>необходимые версии библиотек</b>](./requirements.txt)

3. [<b>Скачать</b>](https://drive.google.com/file/d/1zIrWTQ9XIFITpYqLkuDQx9zOlK3Idzud/view) веса модели

4. Загрузить модель

    ```py
    import torch
    from torchvision import models
    from torch import nn
    from torchvision.transforms import v2
    # Check if CUDA available
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize resnet without pretrained weights
    resnext = models.resnext50_32x4d()
    # Change last layer for current task
    fc_input = resnext.fc.in_features
    resnext.fc = nn.Linear(fc_input, 5, bias=True)
    # Load weights
    resnext.load_state_dict(torch.load('./model.pt'))
    # Move model to GPU if available
    resnext = resnext.to(DEVICE)
    ```

5. Сделать предсказания (пример для одного изображения, для батча изображений лучше использовать *DataLoader*)
    ```py
    # Resnet input size
    NN_INPUT = (224, 224)
    base_transforms = v2.Compose([
        # Resize to match resnet input
        v2.Resize(NN_INPUT, antialias=True),
        # uint8 -> float32
        v2.ToDtype(torch.float32, scale=True),
        # Normalization with ImageNet mean and std
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Read image, move to GPU
    path_to_image = r'YOUR_IMAGE_PATH'
    img = io.read_image(path_to_image).to(DEVICE)
    # Use transforms, reshape [C, H, W] -> [B, C, H, W],
    # where B - batch size, C - channels, H - height, W - width
    img_transformed = base_transforms(img).expand([1, -1, -1, -1])
    # Dict to inverse transform outputs from net
    inverse_transform_dict = {
        0: 'jeff_bezos',
        1: 'bill_gates',
        2: 'steve_jobs',
        3: 'mark_zuckerberg',
        4: 'elon_musk'
    }
    # Get prediction without grad
    with torch.no_grad():
        outputs = resnext(img_transformed).argmax().cpu().item()
    result = inverse_transform_dict[outputs]
    print(f'Predicted label: {result}')
    ```