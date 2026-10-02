import { describe, it, expect } from "vitest";
import { assetUrl, apiUrl, normalizeApiBase, comparisonOptions, combinePreviews, previewFor } from "../multilink/source.js";

describe("Pages and hosted dynamics", () => {
  it("loads project assets under the repository prefix", () => {
    expect(assetUrl("previews/manifest.json", "/my-demo/", "https://yi.github.io/my-demo/")).toBe("https://yi.github.io/my-demo/previews/manifest.json");
    expect(assetUrl("previews/manifest.json", "/", "https://demo.example/")).toBe("https://demo.example/previews/manifest.json");
  });
  it("separates the hosted API origin from the Pages path", () => {
    expect(apiUrl("rollout", "https://dynamics.example/", "/repo/", "https://yi.github.io/repo/")).toBe("https://dynamics.example/api/rollout");
    expect(apiUrl("checkpoints", "", "/repo/", "https://yi.github.io/repo/")).toBe("https://yi.github.io/repo/api/checkpoints");
  });
  it("normalizes a backend URL and rejects unsuitable remote URLs", () => {
    expect(normalizeApiBase("https://api.example/prefix", "https://yi.github.io/")).toBe("https://api.example/prefix/");
    expect(() => normalizeApiBase("http://api.example", "https://yi.github.io/")).toThrow("HTTPS");
    expect(() => normalizeApiBase("https://user:secret@api.example", "https://yi.github.io/")).toThrow("credentials");
  });
  it("restricts saved comparisons to the same variant", () => {
    const a={id:"a",family:"spline",comparison_group:"one"},b={id:"b",family:"spline",comparison_group:"one"},c={id:"c",family:"spline",comparison_group:"two"};
    expect(comparisonOptions([a,b,c],a,"preview")).toEqual([b]);
    expect(comparisonOptions([a,b,c],a,"live")).toEqual([b,c]);
  });
  it("rejects overlays with different physical targets", () => {
    const a={family:"spline",group:"test",task_index:0,runs:[{target_body:[[0,0]],target_pose:[0,0,0],world:[[[0,0]]]}]};
    const b=structuredClone(a);b.runs[0].target_pose[0]=1;
    expect(()=>combinePreviews(a,b)).toThrow("targets");
    expect(combinePreviews(a,structuredClone(a)).runs).toHaveLength(2);
  });
  it("resolves the selected saved task", () => {
    const item={checkpoint:"a",group:"hard_test",task_index:0,file:"one.json"};
    expect(previewFor({previews:[item]},item)).toBe(item);
    expect(()=>previewFor({previews:[item]},{...item,task_index:1})).toThrow("unavailable");
  });
});
