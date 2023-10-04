// {'file_name': '17257760-05a1-11ee-9e2f-0242ac110003.png', 'file_hash': '03aa65118ec6442403b751519cf5677e05118a023565c4a0f36419d30fc8d722', 'file_path': 'images/17257760-05a1-11ee-9e2f-0242ac110003.png', 'file_archive': 'pixel-art-000001.zip', 'image_type': '.png', 'image_width': 384, 'image_height': 384, 'image_size': 4154, 'tags': []}
export interface PhotoInterfaces {
    id: number;
    file_name: string;
    file_hash: string;
    file_path: string;
    file_archive: string;
    image_type: string;
    image_width: number;
    image_height: number;
    image_size: number;
    tags: string[];
}

export class Photo implements PhotoInterfaces {
    id: number;
    file_name: string;
    file_hash: string;
    file_path: string;
    file_archive: string;
    image_type: string;
    image_width: number;
    image_height: number;
    image_size: number;
    tags: string[];

    constructor(photo: PhotoInterfaces) {
        this.id = photo.id;
        this.file_name = photo.file_name;
        this.file_hash = photo.file_hash;
        this.file_path = photo.file_path;
        this.file_archive = photo.file_archive;
        this.image_type = photo.image_type;
        this.image_width = photo.image_width;
        this.image_height = photo.image_height;
        this.image_size = photo.image_size;
        this.tags = photo.tags;
    }

    public getName(): string {
        return this.file_name.split('.')[0];
    }

    public getSize(): string {
        let size = this.image_size;
        let unit = 'bytes';
        if (size > 1024) {
            size = size / 1024;
            unit = 'kb';
        } else if (size > 1024 * 1024) {
            size = size / (1024 * 1024);
            unit = 'mb';
        } else if (size > 1024 * 1024 * 1024) {
            size = size / (1024 * 1024 * 1024);
            unit = 'gb';
        }

        return size.toFixed(2) + ' ' + unit;
    }

    public getArchive(): string {
        return this.file_archive.split('.')[0];
    }

    public getDimensions(): string {
        return this.image_width + ' x ' + this.image_height;
    }

    getType(): string {
        return this.image_type.split('.')[1].toUpperCase();
    }
}