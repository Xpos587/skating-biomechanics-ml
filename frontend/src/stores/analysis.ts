import { create } from "zustand"

export interface AnalysisState {
  currentFrame: number
  isPlaying: boolean
  playbackSpeed: number
  selectedJoint: number | null

  // Actions
  setCurrentFrame: (frame: number) => void
  setIsPlaying: (playing: boolean) => void
  setPlaybackSpeed: (speed: number) => void
  setSelectedJoint: (joint: number | null) => void
  reset: () => void
}

export const useAnalysisStore = create<AnalysisState>(set => ({
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1.0,
  selectedJoint: null,

  setCurrentFrame: frame => set({ currentFrame: frame }),
  setIsPlaying: playing => set({ isPlaying: playing }),
  setPlaybackSpeed: speed => set({ playbackSpeed: speed }),
  setSelectedJoint: joint => set({ selectedJoint: joint }),

  reset: () =>
    set({
      currentFrame: 0,
      isPlaying: false,
      playbackSpeed: 1.0,
      selectedJoint: null,
    }),
}))
