import {Link, useLocation} from "react-router-dom";
import {Home, Icon} from 'react-feather';
import {useEffect, useState} from "react";

interface NavItemProps {
    to: string;
    label: string;
    icon: Icon;
}

const NavItem = ({to, label, icon: Icon}: NavItemProps) => {
    const location = useLocation();
    const isActive = location.pathname === to;

    return (
        <Link to={to}
              className="relative group flex items-center space-x-2 p-2 rounded-full transition-colors duration-200 ease-in-out hover:bg-gray-100"
              style={{backgroundColor: isActive ? "rgba(72,80,222,0.35)" : "#F3F4F6"}}
        >
            <div className={`p-2 rounded-full flex items-center justify-center`}>
                <Icon color="black"/>
            </div>
            <span className="text-black pr-2">{label}</span>
        </Link>
    );
}

let lastScrollTop = 0;
export default function Index() {
    // const [lastScrollTop, setLastScrollTop] = useState<number>(0);
    const [isVisible, setIsVisible] = useState<boolean>(true);

    useEffect(() => {
        const handleScroll = () => {
            const currentScrollTop = window.scrollY;
            setIsVisible(!(currentScrollTop > lastScrollTop));
            lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
        };

        window.addEventListener('scroll', handleScroll, {passive: true});

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <div
            className={`bg-base-100 shadow-xl py-4 px-16 flex justify-center items-center transition-all duration-300 w-full sticky top-0 z-10 ${isVisible ? 'translate-y-0' : '-translate-y-full'}`}
            style={{transition: 'transform 0.3s ease-in-out'}}
        >
            <div className="flex space-x-4 justify-center items-center">
                <NavItem to="/" label="Scrolling" icon={Home}/>
            </div>
        </div>
    );
}
