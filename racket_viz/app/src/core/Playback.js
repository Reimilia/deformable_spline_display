/**
 * Drives all four panels from a single time cursor over a scenario's frames.
 * Mirrors dzhanibekov_display.py's playback model: plays once, then stops (no
 * looping) -- narrative segments are meant to be watched, not looped.
 */
export class Playback {
  /**
   * @param {number[]} times frame timestamps, strictly increasing
   */
  constructor(times) {
    this.times = times;
    this.currentTime = times.length ? times[0] : 0;
    this.playing = false;
  }

  /** Index of the last frame with times[i] <= t, clamped to [0, times.length-1]. */
  frameIndexAtTime(t) {
    const n = this.times.length;
    if (t <= this.times[0]) return 0;
    if (t >= this.times[n - 1]) return n - 1;

    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (this.times[mid] <= t) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    return lo;
  }

  play() {
    const maxT = this.times[this.times.length - 1];
    if (this.currentTime < maxT) {
      this.playing = true;
    }
  }

  pause() {
    this.playing = false;
  }

  seek(t) {
    const maxT = this.times[this.times.length - 1];
    const minT = this.times[0];
    this.currentTime = Math.min(maxT, Math.max(minT, t));
  }

  /** Advance by dt seconds of wall-clock playback; returns false once the
   * scenario has finished (matches the "plays once, then stops" behavior). */
  tick(dt) {
    if (!this.playing) return false;

    const maxT = this.times[this.times.length - 1];
    this.currentTime += dt;
    if (this.currentTime >= maxT) {
      this.currentTime = maxT;
      this.playing = false;
      return false;
    }
    return true;
  }
}
