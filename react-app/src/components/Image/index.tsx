import {Photo, PhotoInterfaces} from "../../model/photo.ts";
import {api} from "../../App.tsx";

interface PhotoItemProps {
    photo: PhotoInterfaces;
    showDescription?: boolean;
}

const ImageItem: React.FC<PhotoItemProps> = ({photo, showDescription = false}) => {
    const _photo = new Photo(photo);


    return (
        <div className="g-gray-100 rounded-lg shadow-md mb-3">
            <div
                className="absolute bottom-3 left-0 right-0 p-2 bg-gray-900 bg-opacity-50 text-white text-xs rounded-lg"
                style={{display: showDescription ? 'block' : 'none'}}
            >
                <div><b>{_photo.task_type}</b></div>
                <div>{_photo.dataset}</div>
                <div>{_photo.image_path}</div>
            </div>
            <div className="bg-gray-100 rounded-lg">
                <img
                    src={`${api.baseUrl}/get-image-data-by-filepath?file_path=${_photo.image_path}`}
                    width={512}
                    height={512}
                    loading="lazy"
                    className="max-w-full rounded-lg"
                    alt={photo.image_hash}
                />
            </div>
        </div>
    );
}

export default ImageItem;