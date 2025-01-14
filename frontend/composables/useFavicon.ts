import { ref } from 'vue';

export function useFavicon() {
  const favicon = ref<HTMLLinkElement | null>(null);
  let blinkInterval: NodeJS.Timeout | null = null;
  let fadeTimeout: NodeJS.Timeout | null = null;

  // Initialize favicon reference
  const initFavicon = () => {
    favicon.value = document.querySelector("link[rel*='icon']");
    if (!favicon.value) {
      favicon.value = document.createElement('link');
      favicon.value.type = 'image/x-icon';
      favicon.value.rel = 'shortcut icon';
      document.head.appendChild(favicon.value);
    }
  };

  // Create dynamic favicon
  const createFavicon = (color: string) => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Clear canvas with black background
    // ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 32, 32);

    // Set line width and stroke style
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;

    // Calculate circle size based on opacity for zoom effect
    const opacityMatch = color.match(/[\d.]+\)$/);
    const opacity = opacityMatch ? parseFloat(opacityMatch[0]) : 1;
    const baseRadius = 7;
    const radius = baseRadius + (1 - opacity) * 1.5; // Subtle zoom out when fading

    // Center the circles more
    const centerY = 16;
    const spacing = 10;
    const centerX = 16;

    // Draw first O
    ctx.beginPath();
    ctx.arc(centerX - spacing/2, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw second O
    ctx.beginPath();
    ctx.arc(centerX + spacing/2, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();

    return canvas.toDataURL();
  };

  // Start processing animation
  const startProcessing = () => {
    initFavicon();
    if (!favicon.value) return;

    let opacity = 1;
    let increasing = false;

    // Clear any existing intervals
    if (blinkInterval) clearInterval(blinkInterval);
    if (fadeTimeout) clearTimeout(fadeTimeout);

    // Create blinking effect
    blinkInterval = setInterval(() => {
      if (increasing) {
        opacity += 0.1;
        if (opacity >= 1) {
          opacity = 1;
          increasing = false;
        }
      } else {
        opacity -= 0.1;
        if (opacity <= 0.3) {
          opacity = 0.3;
          increasing = true;
        }
      }

      const color = `rgba(200, 0, 0, ${opacity})`; // dark red
      if (favicon.value) {
        favicon.value.href = createFavicon(color);
      }
    }, 50);
  };

  // Complete processing animation
  const completeProcessing = () => {
    if (!favicon.value) return;

    // Clear blinking interval
    if (blinkInterval) {
      clearInterval(blinkInterval);
      blinkInterval = null;
    }

    // Show success flash
    favicon.value.href = createFavicon('#00ff00'); // bright green

    // Fade to black
    let opacity = 1;
    if (fadeTimeout) clearTimeout(fadeTimeout);
    
    fadeTimeout = setTimeout(() => {
      const fadeInterval = setInterval(() => {
        opacity -= 0.1;
        if (opacity <= 0) {
          clearInterval(fadeInterval);
          // Reset to original favicon
          if (favicon.value) {
            favicon.value.href = createFavicon('#ffffff'); // white
          }
          return;
        }
        const color = `rgba(0, 255, 0, ${opacity})`;
        if (favicon.value) {
          favicon.value.href = createFavicon(color);
        }
      }, 50);
    }, 500); // Wait before starting fade
  };

  return {
    startProcessing,
    completeProcessing
  };
}
