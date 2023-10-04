import {Photo, PhotoInterfaces} from "../../model/photo.ts";
import {api} from "../../App.tsx";
import {useEffect, useState} from "react";

interface PhotoItemProps {
    photo: PhotoInterfaces;
    showDescription?: boolean;
}

const ImageItem: React.FC<PhotoItemProps> = ({photo, showDescription = false}) => {
    const _photo = new Photo(photo);
    const [base64Image, setBase64Image] = useState<string>('');

    async function getBase64Image() {
        try {
            // @mahdi: replace this endpoint with the enpoint to get base64 from an image hash
            const {data} = await api.get(`/get-image-data`,
                {
                    path: photo.file_name
                }
            )
            setBase64Image(data);
        } catch (error) {
            console.error("Error fetching archives:", error);
        }
    }

    useEffect(() => {
        getBase64Image().then(_ => _);
    }, []);

    return (
        <div className="g-gray-100 rounded-lg shadow-md mb-3">
            <div
                className="absolute bottom-3 left-0 right-0 p-2 bg-gray-900 bg-opacity-50 text-white text-xs rounded-lg"
                style={{display: showDescription ? 'block' : 'none'}}
            >
                <div>{_photo.getName()}</div>
                <div>{_photo.getArchive()}</div>
                <strong className="flex justify-between">
                    <div>{_photo.getDimensions() + " - " + _photo.getType()}</div>
                    <div>{_photo.getSize()}</div>
                </strong>
            </div>
            <div className="bg-gray-100 rounded-lg">
                <img
                    src={`data:image/png;base64,${base64Image}`}
                    width={photo.image_width}
                    height={photo.image_height}
                    loading="lazy"
                    className="max-w-full rounded-lg"
                    alt={photo.file_name}
                />
            </div>
        </div>
    );
}

export default ImageItem;