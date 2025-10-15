"use client";
import { useState, useRef } from "react";
import axios from "axios";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(6.5);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState<string>("");
  const [jobId, setJobId] = useState<string>();
  const [imgUrl, setImgUrl] = useState<string>();
  const [status, setStatus] = useState<string>("idle");
  const pollRef = useRef<any>(null);

  function getApiBase() {
    return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  }

  async function onGenerate() {
    if (!prompt.trim()) return;
    setStatus("submitting");
    setImgUrl(undefined);

    const api = getApiBase();
    const payload: any = {
      prompt,
      negative_prompt: negative,
      steps,
      cfg,
      width,
      height,
    };
    if (seed.trim() !== "") {
      const s = Number(seed);
      if (!Number.isNaN(s)) payload.seed = s;
    }

    const { data } = await axios.post(`${api}/generate`, payload);
    setJobId(data.job_id);
    setStatus("queued");

    pollRef.current = setInterval(async () => {
      const res = await axios.get(`${api}/jobs/${data.job_id}`);
      const s = res.data.status;
      if (s === "done") {
        setStatus("done");
        setImgUrl(`${api}${res.data.output.image_path}`);
        clearInterval(pollRef.current);
      } else if (s === "error" || s === "blocked" || s === "not_found") {
        setStatus(s);
        alert(s === "blocked" ? "Prompt blocked by safety filter" : "Generation error");
        clearInterval(pollRef.current);
      } else {
        setStatus(s);
      }
    }, 1000);
  }

  function disabled() {
    return status === "submitting" || status === "queued" || !prompt.trim();
  }

  return (
    <main className="max-w-3xl mx-auto p-8 space-y-6">
      <h1 className="text-3xl font-bold">Text → Image</h1>

      <section className="space-y-2">
        <label className="block font-medium">Prompt</label>
        <textarea
          className="w-full p-3 border rounded"
          rows={4}
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="A cinematic photo of a neon-lit rainy street with reflections…"
        />
      </section>

      <section className="space-y-2">
        <label className="block font-medium">Negative prompt (optional)</label>
        <input
          className="w-full p-3 border rounded"
          value={negative}
          onChange={e => setNegative(e.target.value)}
          placeholder="blurry, low-res, watermark, text, extra fingers"
        />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block font-medium">Steps: {steps}</label>
          <input
            type="range" min={10} max={60} value={steps}
            onChange={e => setSteps(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block font-medium">CFG: {cfg}</label>
          <input
            type="range" step={0.5} min={1} max={15} value={cfg}
            onChange={e => setCfg(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block font-medium">Width: {width}</label>
          <input
            type="range" step={64} min={512} max={1536} value={width}
            onChange={e => setWidth(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block font-medium">Height: {height}</label>
          <input
            type="range" step={64} min={512} max={1536} value={height}
            onChange={e => setHeight(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="md:col-span-2">
          <label className="block font-medium">Seed (blank for random)</label>
          <input
            className="w-full p-3 border rounded"
            inputMode="numeric"
            value={seed}
            onChange={e => setSeed(e.target.value)}
            placeholder="e.g. 123456"
          />
        </div>
      </section>

      <div className="flex gap-3">
        <button
          className="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
          onClick={onGenerate}
          disabled={disabled()}
        >
          {status === "queued" ? "Generating..." : "Generate"}
        </button>
        <span className="self-center text-sm text-gray-600">Status: {status}</span>
      </div>

      {imgUrl && (
        <div className="mt-4">
          <img src={imgUrl} alt="result" className="rounded" />
          <a href={imgUrl} download className="underline block mt-2">Download</a>
        </div>
      )}
    </main>
  );
}