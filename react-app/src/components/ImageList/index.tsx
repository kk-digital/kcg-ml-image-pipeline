import {useState, useEffect, useRef, useCallback} from "react";
import {Loader} from "react-feather";
import {PhotoInterfaces} from "../../model/photo.ts";
import {Masonry} from "@mui/lab";
import ImageItem from "../Image";
import {api} from "../../App.tsx";

interface ImageListProps {
    query?: { [key: string]: string };
    showDescription?: boolean;
    onSelectPhotos?: (selectedPhotos: PhotoInterfaces[]) => void; // Modificado para retornar array
    dataset?: string | null;
}

export default function ImageList({query, showDescription = false, onSelectPhotos, dataset}: ImageListProps) {
    const LIMIT = 100;

    const [photos, setPhotos] = useState<PhotoInterfaces[]>([]);
    const [page, setPage] = useState<number>(1);
    const [hasMore, setHasMore] = useState<boolean>(true);
    const [numColumns, setNumColumns] = useState<number>(
        Math.floor((window.innerWidth - (window.innerWidth * 0.15)) / 252)
    );
    const [selectedPhotos, setSelectedPhotos] = useState<PhotoInterfaces[]>([]);

    const observerRef = useRef<HTMLDivElement | null>(null);

    const updateNumColumns = useCallback(() => {
        const columns = Math.floor(
            (window.innerWidth - (window.innerWidth * 0.15)) / 266
        );
        if (numColumns !== columns) {
            setNumColumns(columns);
        }
    }, []);

    useEffect(() => {
        updateNumColumns();
        window.addEventListener('resize', updateNumColumns);

        return () => {
            window.removeEventListener('resize', updateNumColumns);
        };
    }, [updateNumColumns]);

    const observer = new IntersectionObserver(
        (entries) => {
            if (
                entries[0].isIntersecting &&
                hasMore
            ) {
                console.log("Update page")
                setPage((prev) => prev + 1);
            }
        },
        {
            rootMargin: '0px',
            threshold: 1.0,
        }
    );

    let getImages = async () => {
        //@mahdi: replace this endpoint with the endpoint to get images metadata
        //Also we may need to change the model, based in the new json response
        try {
            const {data} = await api.get(
                `/image/list-metadata`,
                {
                    dataset: dataset,
                    offset: (page - 1) * LIMIT,
                    limit: LIMIT,
                    ...query,
                }
            )
            if (data.length === 0) {
                setHasMore(false);
            } else {
                setPhotos((prevPhotos) => [...prevPhotos, ...data]);
            }
        } catch (error) {
            console.error("Error fetching photos:", error);
        }
    };


    useEffect(() => {
        setPhotos([]);
        setPage(1);
        setHasMore(true)
    }, [query, dataset]);


    useEffect(() => {
        getImages().then(_ => _);
    }, [page]);

    useEffect(() => {
        if (photos.length === 0 && page === 1) {
            getImages().then(_ => _);
        }
    }, [photos]);

    useEffect(() => {
        if (observerRef.current) {
            observer.observe(observerRef.current);
        }

        return () => {
            if (observerRef.current) {
                observer.unobserve(observerRef.current);
            }
        }
    }, []);


    const handlePhotoClick = (photo: PhotoInterfaces) => {
        if (onSelectPhotos) {
            if (selectedPhotos.includes(photo)) {
                const newSelectedPhotos = selectedPhotos.filter(p => p !== photo);
                setSelectedPhotos(newSelectedPhotos);
                onSelectPhotos(newSelectedPhotos);
            } else {
                const newSelectedPhotos = [...selectedPhotos, photo];
                setSelectedPhotos(newSelectedPhotos);
                onSelectPhotos(newSelectedPhotos);
            }
        }
    };


    return (
        <div className="px-5 py-3">
            <Masonry columns={numColumns}>
                {photos.map((photo: PhotoInterfaces) => (
                    <div
                        key={photo.image_hash}
                        className={`relative ${selectedPhotos.includes(photo) ? 'border-2 border-blue-500' : ''}`}
                        onClick={() => handlePhotoClick(photo)}
                    >
                        {onSelectPhotos && (
                            <div className="absolute top-1 right-1">
                                <input type="checkbox" checked={selectedPhotos.includes(photo)} readOnly
                                       className="cursor-pointer"/>
                            </div>
                        )}
                        <ImageItem photo={photo} showDescription={showDescription}/>
                    </div>
                ))}
            </Masonry>
            <div ref={observerRef} className="flex justify-center items-center mb-3 h-12">
                {hasMore ? <Loader className="animate-spin"/> : 'No more images.'}
            </div>
        </div>
    );
}
