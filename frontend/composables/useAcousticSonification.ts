import { ref } from 'vue';

export function useAcousticSonification() {
  const audioCtx = ref(null);
  const oscillator = ref(null);
  const gainNode = ref(null);
  const isPlaying = ref(false);

  const initAudio = () => {
    if (!audioCtx.value) {
      audioCtx.value = new (window.AudioContext || window.webkitAudioContext)();
      gainNode.value = audioCtx.value.createGain();
      gainNode.value.connect(audioCtx.value.destination);
      gainNode.value.gain.value = 0;
    }
  };

  const playResponse = (freq, amplitude) => {
    initAudio();
    if (audioCtx.value.state === 'suspended') {
      audioCtx.value.resume();
    }

    // Stop previous if any
    if (oscillator.value) {
      oscillator.value.stop();
    }

    oscillator.value = audioCtx.value.createOscillator();
    oscillator.value.type = 'sine';
    oscillator.value.frequency.setValueAtTime(freq, audioCtx.value.currentTime);
    
    oscillator.value.connect(gainNode.value);
    
    const now = audioCtx.value.currentTime;
    // Map simulation amplitude to audio gain
    // Normalize amplitude (assuming it's roughly -1 to 1 or 0 to 1)
    const targetGain = Math.min(0.5, Math.abs(amplitude) * 0.5);
    
    gainNode.value.gain.cancelScheduledValues(now);
    gainNode.value.gain.setValueAtTime(0, now);
    gainNode.value.gain.linearRampToValueAtTime(targetGain, now + 0.1);
    gainNode.value.gain.exponentialRampToValueAtTime(0.001, now + 1.5);

    oscillator.value.start(now);
    oscillator.value.stop(now + 1.6);
  };

  return {
    playResponse,
    isPlaying
  };
}
