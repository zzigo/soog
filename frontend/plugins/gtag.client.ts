export default defineNuxtPlugin((nuxtApp) => {
  const id = 'G-6L0266QCF4';

  if (process.client) {
    // 1. Script injection
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${id}`;
    document.head.appendChild(script);

    // 2. dataLayer initialization
    window.dataLayer = window.dataLayer || [];
    function gtag(...args: any[]) {
      window.dataLayer.push(arguments);
    }
    window.gtag = gtag;

    gtag('js', new Date());
    gtag('config', id);

    // 3. Track route changes
    const router = useRouter();
    router.afterEach((to) => {
      gtag('config', id, {
        page_path: to.fullPath,
        page_location: window.location.origin + to.fullPath
      });
    });
  }
});

// TypeScript support
declare global {
  interface Window {
    dataLayer: any[];
    gtag: (...args: any[]) => void;
  }
}
