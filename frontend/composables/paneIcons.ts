// Stroke icons (24x24 viewBox, 1.6 stroke) for each pod type, shown in the pod header menu.

export const PANE_ICONS: Record<string, string> = {
  auto: 'M4 12h4l2-6 4 12 2-6h4',
  organogram: 'M12 4v5M12 9H6v4M12 9h6v4M6 13v4M18 13v4M10 4h4M4 17h4M16 17h4',
  sound: 'M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2',
  summary: 'M5 6h14M5 10h14M5 14h9M5 18h12',
  geometry: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5',
  acoustic: 'M12 12m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M12 12m-5.5 0a5.5 5.5 0 1 0 11 0a5.5 5.5 0 1 0-11 0M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0',
  sketch: 'M4 20l1-4L16.5 4.5a2 2 0 0 1 3 3L8 19zM14 7l3 3',
  sota: 'M2 12q2.5-9 5 0t5 0 5 0 5 0',
  stage: 'M3 20h18M5 20V10l7-6l7 6v10M9 20v-5h6v5',
};

export const FALLBACK_PANE_ICON = 'M5 5h14v14H5zM5 10h14';

export function paneIcon(key: string): string {
  return PANE_ICONS[key] || FALLBACK_PANE_ICON;
}
