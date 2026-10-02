import { describe, it, expect } from "vitest";
import { bounds, frameAt, validateRollout } from "../multilink/plots.js";

describe("multilink time synchronization and domains",()=>{
  it("preserves equal geometry scale with a rectangular panel",()=>{
    const [x0,x1,y0,y1]=bounds([[0,0],[1,2]],true,400,200);
    expect((x1-x0)/400).toBeCloseTo((y1-y0)/200);
  });
  it("clamps a shorter comparison run at its terminal frame",()=>{
    expect(frameAt([0,.1,.2],10)).toBe(2);
    expect(frameAt([0,.1,.2],.15)).toBe(1);
  });
  it("rejects misaligned metric frames before playback",()=>{
    const r={times:[0,1],body:[[],[]],world:[[],[]],phase_q:[[],[]],phase_p:[[],[]],metrics:{energy:[1,2]}};
    expect(()=>validateRollout({schema_version:1,runs:[r]})).toThrow("Inconsistent metric frames");
  });
});
