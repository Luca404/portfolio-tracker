import { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

interface SwipeNavigationConfig {
  threshold?: number; // Distanza minima in px per attivare lo swipe
  velocityThreshold?: number; // Velocità minima per lo swipe
  routes: string[]; // Array delle route in ordine
}

export function useSwipeNavigation({
  threshold = 120,
  velocityThreshold = 0.3,
  routes
}: SwipeNavigationConfig) {
  const navigate = useNavigate();
  const location = useLocation();
  const touchStartRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const swipeOffsetRef = useRef(0);
  const isSwipingRef = useRef(false);
  const [swipeOffset, setSwipeOffset] = useState(0);
  const [isSwipingHorizontally, setIsSwipingHorizontally] = useState(false);

  useEffect(() => {
    const handleTouchStart = (e: TouchEvent) => {
      // Ignora solo input, select, textarea e elementi con data-no-swipe
      const target = e.target as HTMLElement;
      if (
        target.closest('input') ||
        target.closest('select') ||
        target.closest('textarea') ||
        target.closest('[data-no-swipe]')
      ) {
        return;
      }

      const touch = e.touches[0];
      touchStartRef.current = {
        x: touch.clientX,
        y: touch.clientY,
        time: Date.now()
      };
      swipeOffsetRef.current = 0;
      isSwipingRef.current = false;
      setIsSwipingHorizontally(false);
      setSwipeOffset(0);
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!touchStartRef.current) return;

      const touch = e.touches[0];
      const deltaX = touch.clientX - touchStartRef.current.x;
      const deltaY = touch.clientY - touchStartRef.current.y;

      // Determina se è uno swipe orizzontale o verticale
      if (!isSwipingRef.current && Math.abs(deltaX) > 15) {
        if (Math.abs(deltaX) > Math.abs(deltaY) * 1.5) {
          isSwipingRef.current = true;
          setIsSwipingHorizontally(true);
        }
      }

      // Se è uno swipe orizzontale, aggiorna l'offset e previeni lo scroll
      if (isSwipingRef.current) {
        swipeOffsetRef.current = deltaX;
        setSwipeOffset(deltaX);
        e.preventDefault();
      }
    };

    const handleTouchEnd = () => {
      if (!touchStartRef.current) return;

      const deltaX = swipeOffsetRef.current;
      const deltaTime = Date.now() - touchStartRef.current.time;
      const velocity = Math.abs(deltaX) / deltaTime;

      // Reset visual feedback
      setSwipeOffset(0);
      setIsSwipingHorizontally(false);

      // Controlla se lo swipe è abbastanza lungo o veloce
      if (Math.abs(deltaX) > threshold || velocity > velocityThreshold) {
        const currentIndex = routes.indexOf(location.pathname);

        if (currentIndex !== -1) {
          let nextIndex = -1;

          if (deltaX > 0 && currentIndex > 0) {
            // Swipe verso destra -> pagina precedente
            nextIndex = currentIndex - 1;
          } else if (deltaX < 0 && currentIndex < routes.length - 1) {
            // Swipe verso sinistra -> pagina successiva
            nextIndex = currentIndex + 1;
          }

          if (nextIndex !== -1) {
            navigate(routes[nextIndex]);
          }
        }
      }

      touchStartRef.current = null;
      swipeOffsetRef.current = 0;
      isSwipingRef.current = false;
    };

    // Aggiungi event listeners
    document.addEventListener('touchstart', handleTouchStart, { passive: true });
    document.addEventListener('touchmove', handleTouchMove, { passive: false });
    document.addEventListener('touchend', handleTouchEnd);
    document.addEventListener('touchcancel', handleTouchEnd);

    return () => {
      document.removeEventListener('touchstart', handleTouchStart);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleTouchEnd);
      document.removeEventListener('touchcancel', handleTouchEnd);
    };
  }, [navigate, location.pathname, routes, threshold, velocityThreshold]);

  return { swipeOffset, isSwipingHorizontally };
}
