import ImageList from "../../components/ImageList";
import {useEffect, useState} from "react";
import {api} from "../../App.tsx";
import './style.css'

function ImageFilter(props: {
    scrollDirection: "up" | "down" | null,
    onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void,
    archives: string[],
    checked: boolean,
    onRandChange: () => void
    onShowDescriptionChange: (value: boolean) => void
}) {
    return <div
        className={`fixed bottom-0 left-1/2 transform -translate-x-1/2 p-4 bg-transparent z-50 transition-opacity duration-300 ease-in-out 
                            ${props.scrollDirection === "down" ? "opacity-0 pointer-events-none" : "opacity-100"}`}
    >
        <div className="flex flex-col justify-center items-start shadow-xl bg-white w-fit rounded-xl p-5
            border-2 border-gray-200
        ">
            <div className="flex flex-row gap-2">
                <div className="flex flex-col">
                    <label>Select a dataset:</label>

                    <select
                        className="mb-2 rounded-lg text-lg"
                        style={{backgroundColor: "#F3F4F6"}}
                        onChange={props.onChange}
                    >
                        {props.archives.map(archive => (<option key={archive} value={archive}>{archive}</option>))}
                    </select>
                </div>
                <div className="border-r-2 rounded"></div>
                {/*Rand Option*/}
                <div className="flex flex-col items-start">
                    <label htmlFor="randToggle" className="mr-2">Random:</label>
                    <div
                        className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                        <input
                            type="checkbox"
                            id="randToggle"
                            checked={props.checked}
                            onChange={props.onRandChange}
                            className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-[#A1ACB3] border-4 appearance-none cursor-pointer"
                        />
                        <label htmlFor="randToggle"
                               className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
                    </div>
                </div>
                <div className="border-r-2 rounded"/>
                {/*Show Description*/}
                <div className="flex flex-col items-start">
                    <label htmlFor="showDescriptionToggle" className="mr-2">Description:</label>
                    <div
                        className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                        <input
                            type="checkbox"
                            id="showDescriptionToggle"
                            onChange={(e) => props.onShowDescriptionChange(e.target.checked)}
                            className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-[#A1ACB3] border-4 appearance-none cursor-pointer"
                        />
                        <label htmlFor="showDescriptionToggle"
                               className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
                    </div>
                </div>
            </div>
        </div>
    </div>;
}

export default function Scrolling() {
    const [scrollDirection, setScrollDirection] = useState<'up' | 'down' | null>(null);
    const [lastScrollPos, setLastScrollPos] = useState(0);

    const [datasets, setDatasets] = useState<string[]>([]);
    const [selectedDatasets, setSelectedDatasets] = useState<string | null>(null); // state for selected archive
    const [query, setQuery] = useState({rand: 'false'});
    const [showDescription, setShowDescription] = useState<boolean>(false);


    async function getDatasets() {
        try {
            const {data} = await api.get(`/dataset/list`)
            setDatasets(data);
            setSelectedDatasets(data[0]);
        } catch (error) {
            console.error("Error fetching archives:", error);
        }
    }

    useEffect(() => {
        getDatasets().then(_ => _);
    }, []);

    useEffect(() => {
        const handleScroll = () => {
            const currentScrollPos = window.pageYOffset;

            // Determine scroll direction
            if (currentScrollPos > lastScrollPos) {
                setScrollDirection('down');
            } else if (currentScrollPos < lastScrollPos) {
                setScrollDirection('up');
            }

            setLastScrollPos(currentScrollPos);
        };

        window.addEventListener('scroll', handleScroll);

        // Cleanup event listener on unmount
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, [lastScrollPos]);


    return (
        <div>
            <ImageList
                query={query}
                dataset={selectedDatasets}
                showDescription={showDescription}
            />

            <ImageFilter scrollDirection={scrollDirection}
                         onChange={e => setSelectedDatasets(e.target.value)}
                         archives={datasets}
                         checked={query.rand === 'true'}
                         onRandChange={() => setQuery({rand: query.rand === 'true' ? 'false' : 'true'})}
                         onShowDescriptionChange={(value) => setShowDescription(value)}
            />

        </div>
    )
        ;
}
