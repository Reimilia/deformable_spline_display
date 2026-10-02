import { describe, expect, it } from "vitest";

import { Playback } from "../core/Playback.js";

describe("Playback.frameIndexAtTime", () => {
  const times = [0, 1, 2, 3, 4];

  it("returns the exact frame for an exact timestamp", () => {
    const pb = new Playback(times);
    expect(pb.frameIndexAtTime(2)).toBe(2);
  });

  it("returns the last frame with times[i] <= t for in-between t", () => {
    const pb = new Playback(times);
    expect(pb.frameIndexAtTime(2.9)).toBe(2);
  });

  it("clamps to the first frame for t before the start", () => {
    const pb = new Playback(times);
    expect(pb.frameIndexAtTime(-5)).toBe(0);
  });

  it("clamps to the last frame for t after the end", () => {
    const pb = new Playback(times);
    expect(pb.frameIndexAtTime(999)).toBe(times.length - 1);
  });
});

describe("Playback playback lifecycle", () => {
  it("plays once and stops -- tick() returns false once frames are exhausted", () => {
    const times = [0, 1, 2];
    const pb = new Playback(times);
    pb.play();
    let alive = true;
    for (let i = 0; i < 10 && alive; i++) {
      alive = pb.tick(1.0);
    }
    expect(alive).toBe(false);
    // Must not have advanced past the last frame's time.
    expect(pb.frameIndexAtTime(pb.currentTime)).toBe(times.length - 1);
  });

  it("pause() halts further advancement on tick()", () => {
    const pb = new Playback([0, 1, 2, 3, 4]);
    pb.play();
    pb.tick(1.0);
    const t = pb.currentTime;
    pb.pause();
    pb.tick(1.0);
    expect(pb.currentTime).toBe(t);
  });

  it("seek() jumps the cursor directly", () => {
    const pb = new Playback([0, 1, 2, 3, 4]);
    pb.seek(2.5);
    expect(pb.frameIndexAtTime(pb.currentTime)).toBe(2);
  });
});
