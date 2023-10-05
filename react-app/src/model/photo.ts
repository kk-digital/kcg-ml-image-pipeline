// {'file_name': '17257760-05a1-11ee-9e2f-0242ac110003.png', 'file_hash': '03aa65118ec6442403b751519cf5677e05118a023565c4a0f36419d30fc8d722', 'file_path': 'images/17257760-05a1-11ee-9e2f-0242ac110003.png', 'file_archive': 'pixel-art-000001.zip', 'image_type': '.png', 'image_width': 384, 'image_height': 384, 'image_size': 4154, 'tags': []}
export interface PhotoInterfaces {
    dataset: number;
    task_type: string;
    image_path: string;
    image_hash: string;
}

export class Photo implements PhotoInterfaces {
    dataset: number;
    task_type: string;
    image_path: string;
    image_hash: string;

    constructor(data: PhotoInterfaces) {
        this.dataset = data.dataset;
        this.task_type = data.task_type;
        this.image_path = data.image_path;
        this.image_hash = data.image_hash;
    }
}