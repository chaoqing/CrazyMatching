var kg=Object.defineProperty;var Tg=(e,t,r)=>t in e?kg(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var Js=(e,t,r)=>Tg(e,typeof t!="symbol"?t+"":t,r);/*!
 * ONNX Runtime Web v1.23.0-dev.20250731-e753643480
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */var za=Object.defineProperty,Ig=Object.getOwnPropertyDescriptor,Eg=Object.getOwnPropertyNames,zg=Object.prototype.hasOwnProperty,Cg=(e=>typeof require<"u"?require:typeof Proxy<"u"?new Proxy(e,{get:(t,r)=>(typeof require<"u"?require:t)[r]}):e)(function(e){if(typeof require<"u")return require.apply(this,arguments);throw Error('Dynamic require of "'+e+'" is not supported')}),P=(e,t)=>()=>(e&&(t=e(e=0)),t),Vt=(e,t)=>{for(var r in t)za(e,r,{get:t[r],enumerable:!0})},Ag=(e,t,r,i)=>{if(t&&typeof t=="object"||typeof t=="function")for(let a of Eg(t))!zg.call(e,a)&&a!==r&&za(e,a,{get:()=>t[a],enumerable:!(i=Ig(t,a))||i.enumerable});return e},dr=e=>Ag(za({},"__esModule",{value:!0}),e),Kt,ht,Pt,eo,Bd,Nd=P(()=>{Kt=new Map,ht=[],Pt=(e,t,r)=>{if(t&&typeof t.init=="function"&&typeof t.createInferenceSessionHandler=="function"){let i=Kt.get(e);if(i===void 0)Kt.set(e,{backend:t,priority:r});else{if(i.priority>r)return;if(i.priority===r&&i.backend!==t)throw new Error(`cannot register backend "${e}" using priority ${r}`)}if(r>=0){let a=ht.indexOf(e);a!==-1&&ht.splice(a,1);for(let n=0;n<ht.length;n++)if(Kt.get(ht[n]).priority<=r){ht.splice(n,0,e);return}ht.push(e)}return}throw new TypeError("not a valid backend")},eo=async e=>{let t=Kt.get(e);if(!t)return"backend not found.";if(t.initialized)return t.backend;if(t.aborted)return t.error;{let r=!!t.initPromise;try{return r||(t.initPromise=t.backend.init(e)),await t.initPromise,t.initialized=!0,t.backend}catch(i){return r||(t.error=`${i}`,t.aborted=!0),t.error}finally{delete t.initPromise}}},Bd=async e=>{let t=e.executionProviders||[],r=t.map(d=>typeof d=="string"?d:d.name),i=r.length===0?ht:r,a,n=[],s=new Set;for(let d of i){let p=await eo(d);typeof p=="string"?n.push({name:d,err:p}):(a||(a=p),a===p&&s.add(d))}if(!a)throw new Error(`no available backend found. ERR: ${n.map(d=>`[${d.name}] ${d.err}`).join(", ")}`);for(let{name:d,err:p}of n)r.includes(d)&&console.warn(`removing requested execution provider "${d}" from session options because it is not available: ${p}`);let u=t.filter(d=>s.has(typeof d=="string"?d:d.name));return[a,new Proxy(e,{get:(d,p)=>p==="executionProviders"?u:Reflect.get(d,p)})]}}),Og=P(()=>{Nd()}),Dd,Rg=P(()=>{Dd="1.23.0-dev.20250703-7fc6235861"}),yi,Ie,Md=P(()=>{Rg(),yi="warning",Ie={wasm:{},webgl:{},webgpu:{},versions:{common:Dd},set logLevel(e){if(e!==void 0){if(typeof e!="string"||["verbose","info","warning","error","fatal"].indexOf(e)===-1)throw new Error(`Unsupported logging level: ${e}`);yi=e}},get logLevel(){return yi}},Object.defineProperty(Ie,"logLevel",{enumerable:!0})}),ye,Bg=P(()=>{Md(),ye=Ie}),Ud,Pd,Ng=P(()=>{Ud=(e,t)=>{let r=typeof document<"u"?document.createElement("canvas"):new OffscreenCanvas(1,1);r.width=e.dims[3],r.height=e.dims[2];let i=r.getContext("2d");if(i!=null){let a,n;(t==null?void 0:t.tensorLayout)!==void 0&&t.tensorLayout==="NHWC"?(a=e.dims[2],n=e.dims[3]):(a=e.dims[3],n=e.dims[2]);let s=(t==null?void 0:t.format)!==void 0?t.format:"RGB",u=t==null?void 0:t.norm,d,p;u===void 0||u.mean===void 0?d=[255,255,255,255]:typeof u.mean=="number"?d=[u.mean,u.mean,u.mean,u.mean]:(d=[u.mean[0],u.mean[1],u.mean[2],0],u.mean[3]!==void 0&&(d[3]=u.mean[3])),u===void 0||u.bias===void 0?p=[0,0,0,0]:typeof u.bias=="number"?p=[u.bias,u.bias,u.bias,u.bias]:(p=[u.bias[0],u.bias[1],u.bias[2],0],u.bias[3]!==void 0&&(p[3]=u.bias[3]));let c=n*a,f=0,g=c,_=c*2,w=-1;s==="RGBA"?(f=0,g=c,_=c*2,w=c*3):s==="RGB"?(f=0,g=c,_=c*2):s==="RBG"&&(f=0,_=c,g=c*2);for(let b=0;b<n;b++)for(let S=0;S<a;S++){let v=(e.data[f++]-p[0])*d[0],$=(e.data[g++]-p[1])*d[1],I=(e.data[_++]-p[2])*d[2],T=w===-1?255:(e.data[w++]-p[3])*d[3];i.fillStyle="rgba("+v+","+$+","+I+","+T+")",i.fillRect(S,b,1,1)}if("toDataURL"in r)return r.toDataURL();throw new Error("toDataURL is not supported")}else throw new Error("Can not access image data")},Pd=(e,t)=>{let r=typeof document<"u"?document.createElement("canvas").getContext("2d"):new OffscreenCanvas(1,1).getContext("2d"),i;if(r!=null){let a,n,s;(t==null?void 0:t.tensorLayout)!==void 0&&t.tensorLayout==="NHWC"?(a=e.dims[2],n=e.dims[1],s=e.dims[3]):(a=e.dims[3],n=e.dims[2],s=e.dims[1]);let u=t!==void 0&&t.format!==void 0?t.format:"RGB",d=t==null?void 0:t.norm,p,c;d===void 0||d.mean===void 0?p=[255,255,255,255]:typeof d.mean=="number"?p=[d.mean,d.mean,d.mean,d.mean]:(p=[d.mean[0],d.mean[1],d.mean[2],255],d.mean[3]!==void 0&&(p[3]=d.mean[3])),d===void 0||d.bias===void 0?c=[0,0,0,0]:typeof d.bias=="number"?c=[d.bias,d.bias,d.bias,d.bias]:(c=[d.bias[0],d.bias[1],d.bias[2],0],d.bias[3]!==void 0&&(c[3]=d.bias[3]));let f=n*a;if(t!==void 0&&(t.format!==void 0&&s===4&&t.format!=="RGBA"||s===3&&t.format!=="RGB"&&t.format!=="BGR"))throw new Error("Tensor format doesn't match input tensor dims");let g=4,_=0,w=1,b=2,S=3,v=0,$=f,I=f*2,T=-1;u==="RGBA"?(v=0,$=f,I=f*2,T=f*3):u==="RGB"?(v=0,$=f,I=f*2):u==="RBG"&&(v=0,I=f,$=f*2),i=r.createImageData(a,n);for(let E=0;E<n*a;_+=g,w+=g,b+=g,S+=g,E++)i.data[_]=(e.data[v++]-c[0])*p[0],i.data[w]=(e.data[$++]-c[1])*p[1],i.data[b]=(e.data[I++]-c[2])*p[2],i.data[S]=T===-1?255:(e.data[T++]-c[3])*p[3]}else throw new Error("Can not access image data");return i}}),xr,qd,Wd,Ld,Vd,Gd,Dg=P(()=>{Ca(),xr=(e,t)=>{if(e===void 0)throw new Error("Image buffer must be defined");if(t.height===void 0||t.width===void 0)throw new Error("Image height and width must be defined");if(t.tensorLayout==="NHWC")throw new Error("NHWC Tensor layout is not supported yet");let{height:r,width:i}=t,a=t.norm??{mean:255,bias:0},n,s;typeof a.mean=="number"?n=[a.mean,a.mean,a.mean,a.mean]:n=[a.mean[0],a.mean[1],a.mean[2],a.mean[3]??255],typeof a.bias=="number"?s=[a.bias,a.bias,a.bias,a.bias]:s=[a.bias[0],a.bias[1],a.bias[2],a.bias[3]??0];let u=t.format!==void 0?t.format:"RGBA",d=t.tensorFormat!==void 0&&t.tensorFormat!==void 0?t.tensorFormat:"RGB",p=r*i,c=d==="RGBA"?new Float32Array(p*4):new Float32Array(p*3),f=4,g=0,_=1,w=2,b=3,S=0,v=p,$=p*2,I=-1;u==="RGB"&&(f=3,g=0,_=1,w=2,b=-1),d==="RGBA"?I=p*3:d==="RBG"?(S=0,$=p,v=p*2):d==="BGR"&&($=0,v=p,S=p*2);for(let T=0;T<p;T++,g+=f,w+=f,_+=f,b+=f)c[S++]=(e[g]+s[0])/n[0],c[v++]=(e[_]+s[1])/n[1],c[$++]=(e[w]+s[2])/n[2],I!==-1&&b!==-1&&(c[I++]=(e[b]+s[3])/n[3]);return d==="RGBA"?new Ne("float32",c,[1,4,r,i]):new Ne("float32",c,[1,3,r,i])},qd=async(e,t)=>{let r=typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement,i=typeof ImageData<"u"&&e instanceof ImageData,a=typeof ImageBitmap<"u"&&e instanceof ImageBitmap,n=typeof e=="string",s,u=t??{},d=()=>{if(typeof document<"u")return document.createElement("canvas");if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);throw new Error("Canvas is not supported")},p=c=>typeof HTMLCanvasElement<"u"&&c instanceof HTMLCanvasElement||c instanceof OffscreenCanvas?c.getContext("2d"):null;if(r){let c=d();c.width=e.width,c.height=e.height;let f=p(c);if(f!=null){let g=e.height,_=e.width;if(t!==void 0&&t.resizedHeight!==void 0&&t.resizedWidth!==void 0&&(g=t.resizedHeight,_=t.resizedWidth),t!==void 0){if(u=t,t.tensorFormat!==void 0)throw new Error("Image input config format must be RGBA for HTMLImageElement");u.tensorFormat="RGBA",u.height=g,u.width=_}else u.tensorFormat="RGBA",u.height=g,u.width=_;f.drawImage(e,0,0),s=f.getImageData(0,0,_,g).data}else throw new Error("Can not access image data")}else if(i){let c,f;if(t!==void 0&&t.resizedWidth!==void 0&&t.resizedHeight!==void 0?(c=t.resizedHeight,f=t.resizedWidth):(c=e.height,f=e.width),t!==void 0&&(u=t),u.format="RGBA",u.height=c,u.width=f,t!==void 0){let g=d();g.width=f,g.height=c;let _=p(g);if(_!=null)_.putImageData(e,0,0),s=_.getImageData(0,0,f,c).data;else throw new Error("Can not access image data")}else s=e.data}else if(a){if(t===void 0)throw new Error("Please provide image config with format for Imagebitmap");let c=d();c.width=e.width,c.height=e.height;let f=p(c);if(f!=null){let g=e.height,_=e.width;return f.drawImage(e,0,0,_,g),s=f.getImageData(0,0,_,g).data,u.height=g,u.width=_,xr(s,u)}else throw new Error("Can not access image data")}else{if(n)return new Promise((c,f)=>{let g=d(),_=p(g);if(!e||!_)return f();let w=new Image;w.crossOrigin="Anonymous",w.src=e,w.onload=()=>{g.width=w.width,g.height=w.height,_.drawImage(w,0,0,g.width,g.height);let b=_.getImageData(0,0,g.width,g.height);u.height=g.height,u.width=g.width,c(xr(b.data,u))}});throw new Error("Input data provided is not supported - aborted tensor creation")}if(s!==void 0)return xr(s,u);throw new Error("Input data provided is not supported - aborted tensor creation")},Wd=(e,t)=>{let{width:r,height:i,download:a,dispose:n}=t,s=[1,i,r,4];return new Ne({location:"texture",type:"float32",texture:e,dims:s,download:a,dispose:n})},Ld=(e,t)=>{let{dataType:r,dims:i,download:a,dispose:n}=t;return new Ne({location:"gpu-buffer",type:r??"float32",gpuBuffer:e,dims:i,download:a,dispose:n})},Vd=(e,t)=>{let{dataType:r,dims:i,download:a,dispose:n}=t;return new Ne({location:"ml-tensor",type:r??"float32",mlTensor:e,dims:i,download:a,dispose:n})},Gd=(e,t,r)=>new Ne({location:"cpu-pinned",type:e,data:t,dims:r??[t.length]})}),Tt,nr,_i,Hd,Mg=P(()=>{Tt=new Map([["float32",Float32Array],["uint8",Uint8Array],["int8",Int8Array],["uint16",Uint16Array],["int16",Int16Array],["int32",Int32Array],["bool",Uint8Array],["float64",Float64Array],["uint32",Uint32Array],["int4",Uint8Array],["uint4",Uint8Array]]),nr=new Map([[Float32Array,"float32"],[Uint8Array,"uint8"],[Int8Array,"int8"],[Uint16Array,"uint16"],[Int16Array,"int16"],[Int32Array,"int32"],[Float64Array,"float64"],[Uint32Array,"uint32"]]),_i=!1,Hd=()=>{if(!_i){_i=!0;let e=typeof BigInt64Array<"u"&&BigInt64Array.from,t=typeof BigUint64Array<"u"&&BigUint64Array.from,r=globalThis.Float16Array,i=typeof r<"u"&&r.from;e&&(Tt.set("int64",BigInt64Array),nr.set(BigInt64Array,"int64")),t&&(Tt.set("uint64",BigUint64Array),nr.set(BigUint64Array,"uint64")),i?(Tt.set("float16",r),nr.set(r,"float16")):Tt.set("float16",Uint16Array)}}}),Fd,jd,Ug=P(()=>{Ca(),Fd=e=>{let t=1;for(let r=0;r<e.length;r++){let i=e[r];if(typeof i!="number"||!Number.isSafeInteger(i))throw new TypeError(`dims[${r}] must be an integer, got: ${i}`);if(i<0)throw new RangeError(`dims[${r}] must be a non-negative integer, got: ${i}`);t*=i}return t},jd=(e,t)=>{switch(e.location){case"cpu":return new Ne(e.type,e.data,t);case"cpu-pinned":return new Ne({location:"cpu-pinned",data:e.data,type:e.type,dims:t});case"texture":return new Ne({location:"texture",texture:e.texture,type:e.type,dims:t});case"gpu-buffer":return new Ne({location:"gpu-buffer",gpuBuffer:e.gpuBuffer,type:e.type,dims:t});case"ml-tensor":return new Ne({location:"ml-tensor",mlTensor:e.mlTensor,type:e.type,dims:t});default:throw new Error(`tensorReshape: tensor location ${e.location} is not supported`)}}}),Ne,Ca=P(()=>{Ng(),Dg(),Mg(),Ug(),Ne=class{constructor(e,t,r){Hd();let i,a;if(typeof e=="object"&&"location"in e)switch(this.dataLocation=e.location,i=e.type,a=e.dims,e.location){case"cpu-pinned":{let s=Tt.get(i);if(!s)throw new TypeError(`unsupported type "${i}" to create tensor from pinned buffer`);if(!(e.data instanceof s))throw new TypeError(`buffer should be of type ${s.name}`);this.cpuData=e.data;break}case"texture":{if(i!=="float32")throw new TypeError(`unsupported type "${i}" to create tensor from texture`);this.gpuTextureData=e.texture,this.downloader=e.download,this.disposer=e.dispose;break}case"gpu-buffer":{if(i!=="float32"&&i!=="float16"&&i!=="int32"&&i!=="int64"&&i!=="uint32"&&i!=="uint8"&&i!=="bool"&&i!=="uint4"&&i!=="int4")throw new TypeError(`unsupported type "${i}" to create tensor from gpu buffer`);this.gpuBufferData=e.gpuBuffer,this.downloader=e.download,this.disposer=e.dispose;break}case"ml-tensor":{if(i!=="float32"&&i!=="float16"&&i!=="int32"&&i!=="int64"&&i!=="uint32"&&i!=="uint64"&&i!=="int8"&&i!=="uint8"&&i!=="bool"&&i!=="uint4"&&i!=="int4")throw new TypeError(`unsupported type "${i}" to create tensor from MLTensor`);this.mlTensorData=e.mlTensor,this.downloader=e.download,this.disposer=e.dispose;break}default:throw new Error(`Tensor constructor: unsupported location '${this.dataLocation}'`)}else{let s,u;if(typeof e=="string")if(i=e,u=r,e==="string"){if(!Array.isArray(t))throw new TypeError("A string tensor's data must be a string array.");s=t}else{let d=Tt.get(e);if(d===void 0)throw new TypeError(`Unsupported tensor type: ${e}.`);if(Array.isArray(t)){if(e==="float16"&&d===Uint16Array||e==="uint4"||e==="int4")throw new TypeError(`Creating a ${e} tensor from number array is not supported. Please use ${d.name} as data.`);e==="uint64"||e==="int64"?s=d.from(t,BigInt):s=d.from(t)}else if(t instanceof d)s=t;else if(t instanceof Uint8ClampedArray)if(e==="uint8")s=Uint8Array.from(t);else throw new TypeError("A Uint8ClampedArray tensor's data must be type of uint8");else if(e==="float16"&&t instanceof Uint16Array&&d!==Uint16Array)s=new globalThis.Float16Array(t.buffer,t.byteOffset,t.length);else throw new TypeError(`A ${i} tensor's data must be type of ${d}`)}else if(u=t,Array.isArray(e)){if(e.length===0)throw new TypeError("Tensor type cannot be inferred from an empty array.");let d=typeof e[0];if(d==="string")i="string",s=e;else if(d==="boolean")i="bool",s=Uint8Array.from(e);else throw new TypeError(`Invalid element type of data array: ${d}.`)}else if(e instanceof Uint8ClampedArray)i="uint8",s=Uint8Array.from(e);else{let d=nr.get(e.constructor);if(d===void 0)throw new TypeError(`Unsupported type for tensor data: ${e.constructor}.`);i=d,s=e}if(u===void 0)u=[s.length];else if(!Array.isArray(u))throw new TypeError("A tensor's dims must be a number array");a=u,this.cpuData=s,this.dataLocation="cpu"}let n=Fd(a);if(this.cpuData&&n!==this.cpuData.length&&!((i==="uint4"||i==="int4")&&Math.ceil(n/2)===this.cpuData.length))throw new Error(`Tensor's size(${n}) does not match data length(${this.cpuData.length}).`);this.type=i,this.dims=a,this.size=n}static async fromImage(e,t){return qd(e,t)}static fromTexture(e,t){return Wd(e,t)}static fromGpuBuffer(e,t){return Ld(e,t)}static fromMLTensor(e,t){return Vd(e,t)}static fromPinnedBuffer(e,t,r){return Gd(e,t,r)}toDataURL(e){return Ud(this,e)}toImageData(e){return Pd(this,e)}get data(){if(this.ensureValid(),!this.cpuData)throw new Error("The data is not on CPU. Use `getData()` to download GPU data to CPU, or use `texture` or `gpuBuffer` property to access the GPU data directly.");return this.cpuData}get location(){return this.dataLocation}get texture(){if(this.ensureValid(),!this.gpuTextureData)throw new Error("The data is not stored as a WebGL texture.");return this.gpuTextureData}get gpuBuffer(){if(this.ensureValid(),!this.gpuBufferData)throw new Error("The data is not stored as a WebGPU buffer.");return this.gpuBufferData}get mlTensor(){if(this.ensureValid(),!this.mlTensorData)throw new Error("The data is not stored as a WebNN MLTensor.");return this.mlTensorData}async getData(e){switch(this.ensureValid(),this.dataLocation){case"cpu":case"cpu-pinned":return this.data;case"texture":case"gpu-buffer":case"ml-tensor":{if(!this.downloader)throw new Error("The current tensor is not created with a specified data downloader.");if(this.isDownloading)throw new Error("The current tensor is being downloaded.");try{this.isDownloading=!0;let t=await this.downloader();return this.downloader=void 0,this.dataLocation="cpu",this.cpuData=t,e&&this.disposer&&(this.disposer(),this.disposer=void 0),t}finally{this.isDownloading=!1}}default:throw new Error(`cannot get data from location: ${this.dataLocation}`)}}dispose(){if(this.isDownloading)throw new Error("The current tensor is being downloaded.");this.disposer&&(this.disposer(),this.disposer=void 0),this.cpuData=void 0,this.gpuTextureData=void 0,this.gpuBufferData=void 0,this.mlTensorData=void 0,this.downloader=void 0,this.isDownloading=void 0,this.dataLocation="none"}ensureValid(){if(this.dataLocation==="none")throw new Error("The tensor is disposed.")}reshape(e){if(this.ensureValid(),this.downloader||this.disposer)throw new Error("Cannot reshape a tensor that owns GPU resource.");return jd(this,e)}}}),et,Kd=P(()=>{Ca(),et=Ne}),Ur,wi,tt,je,zt,Ct,Zd=P(()=>{Md(),Ur=(e,t)=>{(typeof Ie.trace>"u"?!Ie.wasm.trace:!Ie.trace)||console.timeStamp(`${e}::ORT::${t}`)},wi=(e,t)=>{var a;let r=((a=new Error().stack)==null?void 0:a.split(/\r\n|\r|\n/g))||[],i=!1;for(let n=0;n<r.length;n++){if(i&&!r[n].includes("TRACE_FUNC")){let s=`FUNC_${e}::${r[n].trim().split(" ")[1]}`;t&&(s+=`::${t}`),Ur("CPU",s);return}r[n].includes("TRACE_FUNC")&&(i=!0)}},tt=e=>{(typeof Ie.trace>"u"?!Ie.wasm.trace:!Ie.trace)||wi("BEGIN",e)},je=e=>{(typeof Ie.trace>"u"?!Ie.wasm.trace:!Ie.trace)||wi("END",e)},zt=e=>{(typeof Ie.trace>"u"?!Ie.wasm.trace:!Ie.trace)||console.time(`ORT::${e}`)},Ct=e=>{(typeof Ie.trace>"u"?!Ie.wasm.trace:!Ie.trace)||console.timeEnd(`ORT::${e}`)}}),Qd,Pg=P(()=>{Nd(),Kd(),Zd(),Qd=class Yd{constructor(t){this.handler=t}async run(t,r,i){tt(),zt("InferenceSession.run");let a={},n={};if(typeof t!="object"||t===null||t instanceof et||Array.isArray(t))throw new TypeError("'feeds' must be an object that use input names as keys and OnnxValue as corresponding values.");let s=!0;if(typeof r=="object"){if(r===null)throw new TypeError("Unexpected argument[1]: cannot be null.");if(r instanceof et)throw new TypeError("'fetches' cannot be a Tensor");if(Array.isArray(r)){if(r.length===0)throw new TypeError("'fetches' cannot be an empty array.");s=!1;for(let p of r){if(typeof p!="string")throw new TypeError("'fetches' must be a string array or an object.");if(this.outputNames.indexOf(p)===-1)throw new RangeError(`'fetches' contains invalid output name: ${p}.`);a[p]=null}if(typeof i=="object"&&i!==null)n=i;else if(typeof i<"u")throw new TypeError("'options' must be an object.")}else{let p=!1,c=Object.getOwnPropertyNames(r);for(let f of this.outputNames)if(c.indexOf(f)!==-1){let g=r[f];(g===null||g instanceof et)&&(p=!0,s=!1,a[f]=g)}if(p){if(typeof i=="object"&&i!==null)n=i;else if(typeof i<"u")throw new TypeError("'options' must be an object.")}else n=r}}else if(typeof r<"u")throw new TypeError("Unexpected argument[1]: must be 'fetches' or 'options'.");for(let p of this.inputNames)if(typeof t[p]>"u")throw new Error(`input '${p}' is missing in 'feeds'.`);if(s)for(let p of this.outputNames)a[p]=null;let u=await this.handler.run(t,a,n),d={};for(let p in u)if(Object.hasOwnProperty.call(u,p)){let c=u[p];c instanceof et?d[p]=c:d[p]=new et(c.type,c.data,c.dims)}return Ct("InferenceSession.run"),je(),d}async release(){return this.handler.dispose()}static async create(t,r,i,a){tt(),zt("InferenceSession.create");let n,s={};if(typeof t=="string"){if(n=t,typeof r=="object"&&r!==null)s=r;else if(typeof r<"u")throw new TypeError("'options' must be an object.")}else if(t instanceof Uint8Array){if(n=t,typeof r=="object"&&r!==null)s=r;else if(typeof r<"u")throw new TypeError("'options' must be an object.")}else if(t instanceof ArrayBuffer||typeof SharedArrayBuffer<"u"&&t instanceof SharedArrayBuffer){let c=t,f=0,g=t.byteLength;if(typeof r=="object"&&r!==null)s=r;else if(typeof r=="number"){if(f=r,!Number.isSafeInteger(f))throw new RangeError("'byteOffset' must be an integer.");if(f<0||f>=c.byteLength)throw new RangeError(`'byteOffset' is out of range [0, ${c.byteLength}).`);if(g=t.byteLength-f,typeof i=="number"){if(g=i,!Number.isSafeInteger(g))throw new RangeError("'byteLength' must be an integer.");if(g<=0||f+g>c.byteLength)throw new RangeError(`'byteLength' is out of range (0, ${c.byteLength-f}].`);if(typeof a=="object"&&a!==null)s=a;else if(typeof a<"u")throw new TypeError("'options' must be an object.")}else if(typeof i<"u")throw new TypeError("'byteLength' must be a number.")}else if(typeof r<"u")throw new TypeError("'options' must be an object.");n=new Uint8Array(c,f,g)}else throw new TypeError("Unexpected argument[0]: must be 'path' or 'buffer'.");let[u,d]=await Bd(s),p=await u.createInferenceSessionHandler(n,d);return Ct("InferenceSession.create"),je(),new Yd(p)}startProfiling(){this.handler.startProfiling()}endProfiling(){this.handler.endProfiling()}get inputNames(){return this.handler.inputNames}get outputNames(){return this.handler.outputNames}get inputMetadata(){return this.handler.inputMetadata}get outputMetadata(){return this.handler.outputMetadata}}}),Xd,qg=P(()=>{Pg(),Xd=Qd}),Wg=P(()=>{}),Lg=P(()=>{}),Vg=P(()=>{}),Gg=P(()=>{}),Hg={};Vt(Hg,{InferenceSession:()=>Xd,TRACE:()=>Ur,TRACE_EVENT_BEGIN:()=>zt,TRACE_EVENT_END:()=>Ct,TRACE_FUNC_BEGIN:()=>tt,TRACE_FUNC_END:()=>je,Tensor:()=>et,env:()=>ye,registerBackend:()=>Pt});var Ue=P(()=>{Og(),Bg(),qg(),Kd(),Wg(),Lg(),Zd(),Vg(),Gg()}),Aa=P(()=>{}),Jd={};Vt(Jd,{default:()=>ep});var bi,$i,ep,Fg=P(()=>{var e;sf(),Bt(),Oa(),bi="ort-wasm-proxy-worker",$i=((e=globalThis.self)==null?void 0:e.name)===bi,$i&&(self.onmessage=t=>{let{type:r,in:i}=t.data;try{switch(r){case"init-wasm":Ra(i.wasm).then(()=>{Qa(i).then(()=>{postMessage({type:r})},a=>{postMessage({type:r,err:a})})},a=>{postMessage({type:r,err:a})});break;case"init-ep":{let{epName:a,env:n}=i;Ya(n,a).then(()=>{postMessage({type:r})},s=>{postMessage({type:r,err:s})});break}case"copy-from":{let{buffer:a}=i,n=Hr(a);postMessage({type:r,out:n});break}case"create":{let{model:a,options:n}=i;Xa(a,n).then(s=>{postMessage({type:r,out:s})},s=>{postMessage({type:r,err:s})});break}case"release":Ja(i),postMessage({type:r});break;case"run":{let{sessionId:a,inputIndices:n,inputs:s,outputIndices:u,options:d}=i;en(a,n,s,u,new Array(u.length).fill(null),d).then(p=>{p.some(c=>c[3]!=="cpu")?postMessage({type:r,err:"Proxy does not support non-cpu tensor location."}):postMessage({type:r,out:p},rn([...s,...p]))},p=>{postMessage({type:r,err:p})});break}case"end-profiling":tn(i),postMessage({type:r});break;default:}}catch(a){postMessage({type:r,err:a})}}),ep=$i?null:t=>new Worker(t??Be,{type:"module",name:bi})}),tp={};Vt(tp,{default:()=>rp});async function to(e={}){var Xs;var t=e,r=typeof window=="object",i=typeof WorkerGlobalScope<"u",a=i&&((Xs=self.name)==null?void 0:Xs.startsWith("em-pthread"));t.mountExternalData=(o,l)=>{o.startsWith("./")&&(o=o.substring(2)),(t.Zc||(t.Zc=new Map)).set(o,l)},t.unmountExternalData=()=>{delete t.Zc};var n=globalThis.SharedArrayBuffer??new WebAssembly.Memory({initial:0,maximum:0,be:!0}).buffer.constructor;let s=o=>async(...l)=>{var h;try{if(t.$c)throw Error("Session already started");let y=t.$c={Nd:l[0],errors:[]},m=await o(...l);if(t.$c!==y)throw Error("Session mismatch");(h=t.gd)==null||h.flush();let k=y.errors;if(0<k.length){let z=await Promise.all(k);if(z=z.filter(N=>N),0<z.length)throw Error(z.join(`
`))}return m}finally{t.$c=null}};t.jsepInit=(o,l)=>{if(o==="webgpu"){[t.gd,t.Dd,t.Hd,t.jd,t.Gd,t.ac,t.Id,t.Kd,t.Ed,t.Fd,t.Jd]=l;let h=t.gd;t.jsepRegisterBuffer=(y,m,k,z)=>h.registerBuffer(y,m,k,z),t.jsepGetBuffer=y=>h.getBuffer(y),t.jsepCreateDownloader=(y,m,k)=>h.createDownloader(y,m,k),t.jsepOnCreateSession=y=>{h.onCreateSession(y)},t.jsepOnReleaseSession=y=>{h.onReleaseSession(y)},t.jsepOnRunStart=y=>h.onRunStart(y),t.Ld=(y,m)=>{h.upload(y,m)}}else if(o==="webnn"){let h=l[0];[t.$d,t.wd,t.webnnEnsureTensor,t.xd,t.webnnDownloadTensor,t.Zd,t.webnnEnableTraceEvent]=l.slice(1),t.webnnReleaseTensorId=t.wd,t.webnnUploadTensor=t.xd,t.webnnRegisterMLContext=t.Zd,t.webnnOnRunStart=y=>h.onRunStart(y),t.webnnOnRunEnd=h.onRunEnd.bind(h),t.webnnOnReleaseSession=y=>{h.onReleaseSession(y)},t.webnnCreateMLTensorDownloader=(y,m)=>h.createMLTensorDownloader(y,m),t.webnnRegisterMLTensor=(y,m,k,z)=>h.registerMLTensor(y,m,k,z),t.webnnCreateMLContext=y=>h.createMLContext(y),t.webnnRegisterMLConstant=(y,m,k,z,N,W)=>h.registerMLConstant(y,m,k,z,N,t.Zc,W),t.webnnRegisterGraphInput=h.registerGraphInput.bind(h),t.webnnIsGraphInput=h.isGraphInput.bind(h),t.webnnRegisterGraphOutput=h.registerGraphOutput.bind(h),t.webnnIsGraphOutput=h.isGraphOutput.bind(h),t.webnnCreateTemporaryTensor=h.createTemporaryTensor.bind(h),t.webnnIsGraphInputOutputTypeSupported=h.isGraphInputOutputTypeSupported.bind(h)}};let u=()=>{let o=l=>(...h)=>{let y=Ye;return h=l(...h),Ye!=y?new Promise((m,k)=>{ai={resolve:m,reject:k}}):h};(()=>{for(let l of["_OrtAppendExecutionProvider","_OrtCreateSession","_OrtRun","_OrtRunWithBinding","_OrtBindInput"])t[l]=o(t[l])})(),s!==void 0&&(t._OrtRun=s(t._OrtRun),t._OrtRunWithBinding=s(t._OrtRunWithBinding)),u=void 0};t.asyncInit=()=>{u==null||u()};var d,p,c=(o,l)=>{throw l},f=import.meta.url,g="";if(r||i){try{g=new URL(".",f).href}catch{}i&&(p=o=>{var l=new XMLHttpRequest;return l.open("GET",o,!1),l.responseType="arraybuffer",l.send(null),new Uint8Array(l.response)}),d=async o=>{if(U(o))return new Promise((h,y)=>{var m=new XMLHttpRequest;m.open("GET",o,!0),m.responseType="arraybuffer",m.onload=()=>{m.status==200||m.status==0&&m.response?h(m.response):y(m.status)},m.onerror=y,m.send(null)});var l=await fetch(o,{credentials:"same-origin"});if(l.ok)return l.arrayBuffer();throw Error(l.status+" : "+l.url)}}var _,w,b,S,v,$,I,T=console.log.bind(console),E=console.error.bind(console),A=T,C=E,O=!1,U=o=>o.startsWith("file://");function x(){G.buffer!=V.buffer&&re()}if(a){let o=function(l){try{var h=l.data,y=h.Vc;if(y==="load"){let m=[];self.onmessage=k=>m.push(k),I=()=>{postMessage({Vc:"loaded"});for(let k of m)o(k);self.onmessage=o};for(let k of h.Ad)t[k]&&!t[k].proxy||(t[k]=(...z)=>{postMessage({Vc:"callHandler",zd:k,args:z})},k=="print"&&(A=t[k]),k=="printErr"&&(C=t[k]));G=h.Wd,re(),$(h.Xd)}else if(y==="run"){(function(m){var k=(x(),q)[m+52>>>2>>>0];m=(x(),q)[m+56>>>2>>>0],ss(k,k-m),se(k)})(h.Tc),pi(h.Tc,0,0,1,0,0),on(),ri(h.Tc),Y||(es(),Y=!0);try{_f(h.Pd,h.dd)}catch(m){if(m!="unwind")throw m}}else h.target!=="setimmediate"&&(y==="checkMailbox"?Y&&gr():y&&(C(`worker: received unknown command ${y}`),C(h)))}catch(m){throw ts(),m}};var Y=!1;self.onunhandledrejection=l=>{throw l.reason||l},self.onmessage=o}var G,V,te,ee,F,R,q,X,_e,D,L,K=!1;function re(){var o=G.buffer;t.HEAP8=V=new Int8Array(o),ee=new Int16Array(o),t.HEAPU8=te=new Uint8Array(o),F=new Uint16Array(o),t.HEAP32=R=new Int32Array(o),t.HEAPU32=q=new Uint32Array(o),X=new Float32Array(o),_e=new Float64Array(o),D=new BigInt64Array(o),L=new BigUint64Array(o)}function ze(){K=!0,a?I():pt.tb()}var Ke,st=0,_t=null;function Pe(){if(--st==0&&_t){var o=_t;_t=null,o()}}function Te(o){throw C(o="Aborted("+o+")"),O=!0,o=new WebAssembly.RuntimeError(o+". Build with -sASSERTIONS for more info."),v==null||v(o),o}function ot(){return{a:{ma:Pm,ib:Um,g:wf,J:bf,f:$f,m:vf,h:xf,ha:Sf,b:kf,T:Tf,Ja:hn,n:If,_:yn,Za:_n,Fa:wn,Ha:bn,_a:$n,Xa:vn,Qa:xn,Wa:Sn,ka:kn,Ga:Tn,Da:In,Ya:En,Ea:zn,db:Ef,ea:zf,ya:Cf,wa:Of,da:Bf,O:Nf,I:Df,xa:Mf,Z:Gf,za:Hf,Ta:Ff,Ba:jf,Ka:Kf,ua:Zf,fa:Qf,Sa:ri,ab:Yf,S:tm,p:sm,c:Jr,jb:om,y:um,N:lm,C:dm,s:pm,r:Mn,kb:Mn,K:cm,R:hm,j:fm,v:mm,q:gm,l:ym,Na:_m,Oa:wm,Pa:bm,La:Wn,Ma:Ln,va:Vn,fb:vm,cb:Sm,u:km,aa:Tm,ga:Im,bb:xm,V:Em,$a:zm,Ca:Cm,F:$m,U:Am,la:$r,Aa:Om,hb:Rm,gb:Bm,Ua:jn,Va:Kn,Ia:Kr,$:Zn,ja:Qn,Ra:Yn,ia:Xn,mb:vg,oa:fg,nb:$g,pa:hg,G:sg,d:Vm,o:Wm,w:qm,B:eg,rb:lg,L:ig,x:Fm,qa:pg,X:mg,ba:ug,ob:wg,pb:_g,qb:dg,ra:og,P:ag,na:bg,Y:cg,e:Hm,z:Gm,k:Lm,lb:xg,t:jm,A:Km,D:Zm,E:Jm,M:tg,sb:ng,Q:gg,ca:rg,W:yg,sa:Ym,ta:Qm,H:Xm,i:Dm,a:G,eb:jr}}}class xe{constructor(l){Js(this,"name","ExitStatus");this.message=`Program terminated with exit(${l})`,this.status=l}}var be=o=>{o.terminate(),o.onmessage=()=>{}},Re=[],cr=o=>{ut.length==0&&(ln(),un(ut[0]));var l=ut.pop();if(!l)return 6;Gt.push(l),wt[o.Tc]=l,l.Tc=o.Tc;var h={Vc:"run",Pd:o.Od,dd:o.dd,Tc:o.Tc};return l.postMessage(h,o.vd),0},Ze=0,we=(o,l,...h)=>{for(var y=2*h.length,m=oe(),k=hi(8*y),z=k>>>3,N=0;N<h.length;N++){var W=h[N];typeof W=="bigint"?((x(),D)[z+2*N>>>0]=1n,(x(),D)[z+2*N+1>>>0]=W):((x(),D)[z+2*N>>>0]=0n,(x(),_e)[z+2*N+1>>>0]=W)}return o=rs(o,0,y,k,l),se(m),o};function jr(o){if(a)return we(0,1,o);if(b=o,!(0<Ze)){for(var l of Gt)be(l);for(l of ut)be(l);ut=[],Gt=[],wt={},O=!0}c(0,new xe(o))}function an(o){if(a)return we(1,0,o);Kr(o)}var Kr=o=>{if(b=o,a)throw an(o),"unwind";jr(o)},ut=[],Gt=[],nn=[],wt={},sn=o=>{var l=o.Tc;delete wt[l],ut.push(o),Gt.splice(Gt.indexOf(o),1),o.Tc=0,is(l)};function on(){nn.forEach(o=>o())}var un=o=>new Promise(l=>{o.onmessage=m=>{var k=(m=m.data).Vc;if(m.ad&&m.ad!=di()){var z=wt[m.ad];z?z.postMessage(m,m.vd):C(`Internal error! Worker sent a message "${k}" to target pthread ${m.ad}, but that thread no longer exists!`)}else k==="checkMailbox"?gr():k==="spawnThread"?cr(m):k==="cleanupThread"?sn(wt[m.Qd]):k==="loaded"?(o.loaded=!0,l(o)):m.target==="setimmediate"?o.postMessage(m):k==="callHandler"?t[m.zd](...m.args):k&&C(`worker sent an unknown command ${k}`)},o.onerror=m=>{throw C(`worker sent an error! ${m.filename}:${m.lineno}: ${m.message}`),m};var h,y=[];for(h of[])t.propertyIsEnumerable(h)&&y.push(h);o.postMessage({Vc:"load",Ad:y,Wd:G,Xd:w})});function ln(){var o=new Worker((()=>{let l=URL;return import.meta.url>"file:"&&import.meta.url<"file;"?new l("ort.bundle.min.mjs",import.meta.url):new URL(import.meta.url)})(),{type:"module",workerData:"em-pthread",name:"em-pthread"});ut.push(o)}var _f=(o,l)=>{Ze=0,o=fi(o,l),0<Ze?b=o:ci(o)},hr=[],fr=0;function wf(o){var l=new Zr(o>>>=0);return(x(),V)[l.Uc+12>>>0]==0&&(dn(l,!0),fr--),pn(l,!1),hr.push(l),us(o),ds(o)}var Dt=0,bf=()=>{ue(0,0);var o=hr.pop();os(o.ed),Dt=0};function dn(o,l){l=l?1:0,(x(),V)[o.Uc+12>>>0]=l}function pn(o,l){l=l?1:0,(x(),V)[o.Uc+13>>>0]=l}class Zr{constructor(l){this.ed=l,this.Uc=l-24}}var Qr=o=>{var l=Dt;if(!l)return jt(0),0;var h=new Zr(l);(x(),q)[h.Uc+16>>>2>>>0]=l;var y=(x(),q)[h.Uc+4>>>2>>>0];if(!y)return jt(0),l;for(var m of o){if(m===0||m===y)break;if(ls(m,y,h.Uc+16))return jt(m),l}return jt(y),l};function $f(){return Qr([])}function vf(o){return Qr([o>>>0])}function xf(o,l,h,y){return Qr([o>>>0,l>>>0,h>>>0,y>>>0])}var Sf=()=>{var o=hr.pop();o||Te("no exception to throw");var l=o.ed;throw(x(),V)[o.Uc+13>>>0]==0&&(hr.push(o),pn(o,!0),dn(o,!1),fr++),Dt=l};function kf(o,l,h){var y=new Zr(o>>>=0);throw l>>>=0,h>>>=0,(x(),q)[y.Uc+16>>>2>>>0]=0,(x(),q)[y.Uc+4>>>2>>>0]=l,(x(),q)[y.Uc+8>>>2>>>0]=h,fr++,Dt=o}var Tf=()=>fr;function cn(o,l,h,y){return a?we(2,1,o,l,h,y):hn(o,l,h,y)}function hn(o,l,h,y){if(o>>>=0,h>>>=0,y>>>=0,n===void 0)return 6;var m=[];return a&&m.length===0?cn(o,l>>>=0,h,y):(o={Od:h,Tc:o,dd:y,vd:m},a?(o.Vc="spawnThread",postMessage(o,m),0):cr(o))}function If(o){throw Dt||(Dt=o>>>0),Dt}var fn=typeof TextDecoder<"u"?new TextDecoder:void 0,mn=(o,l,h,y)=>{if(h=l+h,y)return h;for(;o[l]&&!(l>=h);)++l;return l},gn=(o,l=0,h,y)=>{if(16<(h=mn(o,l>>>=0,h,y))-l&&o.buffer&&fn)return fn.decode(o.buffer instanceof ArrayBuffer?o.subarray(l,h):o.slice(l,h));for(y="";l<h;){var m=o[l++];if(128&m){var k=63&o[l++];if((224&m)==192)y+=String.fromCharCode((31&m)<<6|k);else{var z=63&o[l++];65536>(m=(240&m)==224?(15&m)<<12|k<<6|z:(7&m)<<18|k<<12|z<<6|63&o[l++])?y+=String.fromCharCode(m):(m-=65536,y+=String.fromCharCode(55296|m>>10,56320|1023&m))}}else y+=String.fromCharCode(m)}return y},Se=(o,l,h)=>(o>>>=0)?gn((x(),te),o,l,h):"";function yn(o,l,h){return a?we(3,1,o,l,h):0}function _n(o,l){if(a)return we(4,1,o,l)}function wn(o,l){if(a)return we(5,1,o,l)}function bn(o,l,h){if(a)return we(6,1,o,l,h)}function $n(o,l,h){return a?we(7,1,o,l,h):0}function vn(o,l){if(a)return we(8,1,o,l)}function xn(o,l,h){if(a)return we(9,1,o,l,h)}function Sn(o,l,h,y){if(a)return we(10,1,o,l,h,y)}function kn(o,l,h,y){if(a)return we(11,1,o,l,h,y)}function Tn(o,l,h,y){if(a)return we(12,1,o,l,h,y)}function In(o){if(a)return we(13,1,o)}function En(o,l){if(a)return we(14,1,o,l)}function zn(o,l,h){if(a)return we(15,1,o,l,h)}var Ef=()=>Te(""),Qe=o=>{o>>>=0;for(var l="";;){var h=(x(),te)[o++>>>0];if(!h)return l;l+=String.fromCharCode(h)}},Yr={},Xr={},Mt=class extends Error{constructor(o){super(o),this.name="BindingError"}};function rt(o,l,h={}){return function(y,m,k={}){var z=m.name;if(!y)throw new Mt(`type "${z}" must have a positive integer typeid pointer`);if(Xr.hasOwnProperty(y)){if(k.Bd)return;throw new Mt(`Cannot register type '${z}' twice`)}Xr[y]=m,Yr.hasOwnProperty(y)&&(m=Yr[y],delete Yr[y],m.forEach(N=>N()))}(o,l,h)}var Cn=(o,l,h)=>{switch(l){case 1:return h?y=>(x(),V)[y>>>0]:y=>(x(),te)[y>>>0];case 2:return h?y=>(x(),ee)[y>>>1>>>0]:y=>(x(),F)[y>>>1>>>0];case 4:return h?y=>(x(),R)[y>>>2>>>0]:y=>(x(),q)[y>>>2>>>0];case 8:return h?y=>(x(),D)[y>>>3>>>0]:y=>(x(),L)[y>>>3>>>0];default:throw new TypeError(`invalid integer width (${l}): ${o}`)}};function zf(o,l,h,y,m){o>>>=0,h>>>=0,l=Qe(l>>>0);let k=z=>z;if(y=y===0n){let z=8*h;k=N=>BigInt.asUintN(z,N),m=k(m)}rt(o,{name:l,Pc:k,Xc:(z,N)=>(typeof N=="number"&&(N=BigInt(N)),N),Wc:Cn(l,h,!y),Yc:null})}function Cf(o,l,h,y){rt(o>>>=0,{name:l=Qe(l>>>0),Pc:function(m){return!!m},Xc:function(m,k){return k?h:y},Wc:function(m){return this.Pc((x(),te)[m>>>0])},Yc:null})}var An=[],bt=[0,1,,1,null,1,!0,1,!1,1];function Jr(o){9<(o>>>=0)&&--bt[o+1]==0&&(bt[o]=void 0,An.push(o))}var Me=o=>{if(!o)throw new Mt(`Cannot use deleted val. handle = ${o}`);return bt[o]},qe=o=>{switch(o){case void 0:return 2;case null:return 4;case!0:return 6;case!1:return 8;default:let l=An.pop()||bt.length;return bt[l]=o,bt[l+1]=1,l}};function ei(o){return this.Pc((x(),q)[o>>>2>>>0])}var Af={name:"emscripten::val",Pc:o=>{var l=Me(o);return Jr(o),l},Xc:(o,l)=>qe(l),Wc:ei,Yc:null};function Of(o){return rt(o>>>0,Af)}var Rf=(o,l)=>{switch(l){case 4:return function(h){return this.Pc((x(),X)[h>>>2>>>0])};case 8:return function(h){return this.Pc((x(),_e)[h>>>3>>>0])};default:throw new TypeError(`invalid float width (${l}): ${o}`)}};function Bf(o,l,h){h>>>=0,rt(o>>>=0,{name:l=Qe(l>>>0),Pc:y=>y,Xc:(y,m)=>m,Wc:Rf(l,h),Yc:null})}function Nf(o,l,h,y,m){o>>>=0,h>>>=0,l=Qe(l>>>0);let k=N=>N;if(y===0){var z=32-8*h;k=N=>N<<z>>>z,m=k(m)}rt(o,{name:l,Pc:k,Xc:(N,W)=>W,Wc:Cn(l,h,y!==0),Yc:null})}function Df(o,l,h){function y(k){var z=(x(),q)[k>>>2>>>0];return k=(x(),q)[k+4>>>2>>>0],new m((x(),V).buffer,k,z)}var m=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array,BigInt64Array,BigUint64Array][l];rt(o>>>=0,{name:h=Qe(h>>>0),Pc:y,Wc:y},{Bd:!0})}var lt=(o,l,h)=>{var y=(x(),te);if(l>>>=0,0<h){var m=l;h=l+h-1;for(var k=0;k<o.length;++k){var z=o.codePointAt(k);if(127>=z){if(l>=h)break;y[l++>>>0]=z}else if(2047>=z){if(l+1>=h)break;y[l++>>>0]=192|z>>6,y[l++>>>0]=128|63&z}else if(65535>=z){if(l+2>=h)break;y[l++>>>0]=224|z>>12,y[l++>>>0]=128|z>>6&63,y[l++>>>0]=128|63&z}else{if(l+3>=h)break;y[l++>>>0]=240|z>>18,y[l++>>>0]=128|z>>12&63,y[l++>>>0]=128|z>>6&63,y[l++>>>0]=128|63&z,k++}}y[l>>>0]=0,o=l-m}else o=0;return o},mr=o=>{for(var l=0,h=0;h<o.length;++h){var y=o.charCodeAt(h);127>=y?l++:2047>=y?l+=2:55296<=y&&57343>=y?(l+=4,++h):l+=3}return l};function Mf(o,l){rt(o>>>=0,{name:l=Qe(l>>>0),Pc(h){var y=(x(),q)[h>>>2>>>0];return y=Se(h+4,y,!0),Xe(h),y},Xc(h,y){y instanceof ArrayBuffer&&(y=new Uint8Array(y));var m=typeof y=="string";if(!(m||ArrayBuffer.isView(y)&&y.BYTES_PER_ELEMENT==1))throw new Mt("Cannot pass non-string to std::string");var k=m?mr(y):y.length,z=Ft(4+k+1),N=z+4;return(x(),q)[z>>>2>>>0]=k,m?lt(y,N,k+1):(x(),te).set(y,N>>>0),h!==null&&h.push(Xe,z),z},Wc:ei,Yc(h){Xe(h)}})}var On=typeof TextDecoder<"u"?new TextDecoder("utf-16le"):void 0,Uf=(o,l,h)=>{if(o>>>=1,16<(l=mn((x(),F),o,l/2,h))-o&&On)return On.decode((x(),F).buffer instanceof ArrayBuffer?(x(),F).subarray(o>>>0,l>>>0):(x(),F).slice(o,l));for(h="";o<l;++o){var y=(x(),F)[o>>>0];h+=String.fromCharCode(y)}return h},Pf=(o,l,h)=>{if(h??(h=2147483647),2>h)return 0;var y=l;h=(h-=2)<2*o.length?h/2:o.length;for(var m=0;m<h;++m){var k=o.charCodeAt(m);(x(),ee)[l>>>1>>>0]=k,l+=2}return(x(),ee)[l>>>1>>>0]=0,l-y},qf=o=>2*o.length,Wf=(o,l,h)=>{var y="";o>>>=2;for(var m=0;!(m>=l/4);m++){var k=(x(),q)[o+m>>>0];if(!k&&!h)break;y+=String.fromCodePoint(k)}return y},Lf=(o,l,h)=>{if(l>>>=0,h??(h=2147483647),4>h)return 0;var y=l;h=y+h-4;for(var m=0;m<o.length;++m){var k=o.codePointAt(m);if(65535<k&&m++,(x(),R)[l>>>2>>>0]=k,(l+=4)+4>h)break}return(x(),R)[l>>>2>>>0]=0,l-y},Vf=o=>{for(var l=0,h=0;h<o.length;++h)65535<o.codePointAt(h)&&h++,l+=4;return l};function Gf(o,l,h){if(o>>>=0,l>>>=0,h=Qe(h>>>=0),l===2)var y=Uf,m=Pf,k=qf;else y=Wf,m=Lf,k=Vf;rt(o,{name:h,Pc:z=>{var N=(x(),q)[z>>>2>>>0];return N=y(z+4,N*l,!0),Xe(z),N},Xc:(z,N)=>{if(typeof N!="string")throw new Mt(`Cannot pass non-string to C++ string type ${h}`);var W=k(N),H=Ft(4+W+l);return(x(),q)[H>>>2>>>0]=W/l,m(N,H+4,W+l),z!==null&&z.push(Xe,H),H},Wc:ei,Yc(z){Xe(z)}})}function Hf(o,l){rt(o>>>=0,{Cd:!0,name:l=Qe(l>>>0),Pc:()=>{},Xc:()=>{}})}function Ff(o){pi(o>>>0,!i,1,!r,131072,!1),on()}var ti=o=>{if(!O)try{if(o(),!(0<Ze))try{a?ci(b):Kr(b)}catch(l){l instanceof xe||l=="unwind"||c(0,l)}}catch(l){l instanceof xe||l=="unwind"||c(0,l)}};function ri(o){o>>>=0,typeof Atomics.Vd=="function"&&(Atomics.Vd((x(),R),o>>>2,o).value.then(gr),o+=128,Atomics.store((x(),R),o>>>2,1))}var gr=()=>{var o=di();o&&(ri(o),ti(ns))};function jf(o,l){(o>>>=0)==l>>>0?setTimeout(gr):a?postMessage({ad:o,Vc:"checkMailbox"}):(o=wt[o])&&o.postMessage({Vc:"checkMailbox"})}var yr=[];function Kf(o,l,h,y,m){for(l>>>=0,y/=2,yr.length=y,h=m>>>0>>>3,m=0;m<y;m++)(x(),D)[h+2*m>>>0]?yr[m]=(x(),D)[h+2*m+1>>>0]:yr[m]=(x(),_e)[h+2*m+1>>>0];return(l?li[l]:Mm[o])(...yr)}var Zf=()=>{Ze=0};function Qf(o){o>>>=0,a?postMessage({Vc:"cleanupThread",Qd:o}):sn(wt[o])}function Yf(o){}var _r=o=>{try{o()}catch(l){Te(l)}};function Xf(o){var l=(...h)=>{wr.push(o);try{return o(...h)}finally{O||(wr.pop(),Ye&&dt===1&&wr.length===0&&(dt=0,Ze+=1,_r(Zs),typeof Fibers<"u"&&Fibers.de()))}};return Nn.set(o,l),l}var dt=0,Ye=null,Rn=0,wr=[],ii=new Map,Bn=new Map,Nn=new Map,Jf=0,ai=null,em=[],Dn=o=>function(l){if(!O){if(dt===0){var h=!1,y=!1;l((m=0)=>{if(!O&&(Rn=m,h=!0,y)){dt=2,_r(()=>Qs(Ye)),typeof MainLoop<"u"&&MainLoop.yd&&MainLoop.resume(),m=!1;try{var k=function(){var W=(x(),R)[Ye+8>>>2>>>0];return W=Bn.get(W),W=Nn.get(W),--Ze,W()}()}catch(W){k=W,m=!0}var z=!1;if(!Ye){var N=ai;N&&(ai=null,(m?N.reject:N.resolve)(k),z=!0)}if(m&&!z)throw k}}),y=!0,h||(dt=1,Ye=function(){var m=Ft(65548),k=m+12;if((x(),q)[m>>>2>>>0]=k,(x(),q)[m+4>>>2>>>0]=k+65536,k=wr[0],!ii.has(k)){var z=Jf++;ii.set(k,z),Bn.set(z,k)}return k=ii.get(k),(x(),R)[m+8>>>2>>>0]=k,m}(),typeof MainLoop<"u"&&MainLoop.yd&&MainLoop.pause(),_r(()=>Ks(Ye)))}else dt===2?(dt=0,_r(Ys),Xe(Ye),Ye=null,em.forEach(ti)):Te(`invalid state: ${dt}`);return Rn}}(l=>{o().then(l)});function tm(o){return o>>>=0,Dn(async()=>{var l=await Me(o);return qe(l)})}var ni=[],rm=o=>{var l=ni.length;return ni.push(o),l},im=(o,l)=>{for(var h=Array(o),y=0;y<o;++y){var m=y,k=(x(),q)[l+4*y>>>2>>>0],z=Xr[k];if(z===void 0)throw o=`parameter ${y}`,k=Jn(k),l=Qe(k),Xe(k),new Mt(`${o} has unknown type ${l}`);h[m]=z}return h},am=(o,l,h)=>{var y=[];return o=o(y,h),y.length&&((x(),q)[l>>>2>>>0]=qe(y)),o},nm={},br=o=>{var l=nm[o];return l===void 0?Qe(o):l};function sm(o,l,h){var[y,...m]=im(o,l>>>0);l=y.Xc.bind(y);var k=m.map(W=>W.Wc.bind(W));o--;var z={toValue:Me};switch(o=k.map((W,H)=>{var ne=`argFromPtr${H}`;return z[ne]=W,`${ne}(args${H?"+"+8*H:""})`}),h){case 0:var N="toValue(handle)";break;case 2:N="new (toValue(handle))";break;case 3:N="";break;case 1:z.getStringOrSymbol=br,N="toValue(handle)[getStringOrSymbol(methodName)]"}return N+=`(${o})`,y.Cd||(z.toReturnWire=l,z.emval_returnValue=am,N=`return emval_returnValue(toReturnWire, destructorsRef, ${N})`),N=`return function (handle, methodName, destructorsRef, args) {
  ${N}
  }`,h=new Function(Object.keys(z),N)(...Object.values(z)),N=`methodCaller<(${m.map(W=>W.name)}) => ${y.name}>`,rm(Object.defineProperty(h,"name",{value:N}))}function om(o,l){return l>>>=0,(o=Me(o>>>0))==Me(l)}function um(o){return(o>>>=0)==0?qe(globalThis):(o=br(o),qe(globalThis[o]))}function lm(o){return o=br(o>>>0),qe(t[o])}function dm(o,l){return l>>>=0,o=Me(o>>>0),l=Me(l),qe(o[l])}function pm(o){9<(o>>>=0)&&(bt[o+1]+=1)}function Mn(o,l,h,y,m){return ni[o>>>0](l>>>0,h>>>0,y>>>0,m>>>0)}function cm(){return qe([])}function hm(o){o=Me(o>>>0);for(var l=Array(o.length),h=0;h<o.length;h++)l[h]=o[h];return qe(l)}function fm(o){return qe(br(o>>>0))}function mm(){return qe({})}function gm(o){for(var l=Me(o>>>=0);l.length;){var h=l.pop();l.pop()(h)}Jr(o)}function ym(o,l,h){l>>>=0,h>>>=0,o=Me(o>>>0),l=Me(l),h=Me(h),o[l]=h}function _m(o,l){o=-9007199254740992>o||9007199254740992<o?NaN:Number(o),l>>>=0,o=new Date(1e3*o),(x(),R)[l>>>2>>>0]=o.getUTCSeconds(),(x(),R)[l+4>>>2>>>0]=o.getUTCMinutes(),(x(),R)[l+8>>>2>>>0]=o.getUTCHours(),(x(),R)[l+12>>>2>>>0]=o.getUTCDate(),(x(),R)[l+16>>>2>>>0]=o.getUTCMonth(),(x(),R)[l+20>>>2>>>0]=o.getUTCFullYear()-1900,(x(),R)[l+24>>>2>>>0]=o.getUTCDay(),o=(o.getTime()-Date.UTC(o.getUTCFullYear(),0,1,0,0,0,0))/864e5|0,(x(),R)[l+28>>>2>>>0]=o}var Un=o=>o%4==0&&(o%100!=0||o%400==0),Pn=[0,31,60,91,121,152,182,213,244,274,305,335],qn=[0,31,59,90,120,151,181,212,243,273,304,334];function wm(o,l){o=-9007199254740992>o||9007199254740992<o?NaN:Number(o),l>>>=0,o=new Date(1e3*o),(x(),R)[l>>>2>>>0]=o.getSeconds(),(x(),R)[l+4>>>2>>>0]=o.getMinutes(),(x(),R)[l+8>>>2>>>0]=o.getHours(),(x(),R)[l+12>>>2>>>0]=o.getDate(),(x(),R)[l+16>>>2>>>0]=o.getMonth(),(x(),R)[l+20>>>2>>>0]=o.getFullYear()-1900,(x(),R)[l+24>>>2>>>0]=o.getDay();var h=(Un(o.getFullYear())?Pn:qn)[o.getMonth()]+o.getDate()-1|0;(x(),R)[l+28>>>2>>>0]=h,(x(),R)[l+36>>>2>>>0]=-60*o.getTimezoneOffset(),h=new Date(o.getFullYear(),6,1).getTimezoneOffset();var y=new Date(o.getFullYear(),0,1).getTimezoneOffset();o=0|(h!=y&&o.getTimezoneOffset()==Math.min(y,h)),(x(),R)[l+32>>>2>>>0]=o}function bm(o){o>>>=0;var l=new Date((x(),R)[o+20>>>2>>>0]+1900,(x(),R)[o+16>>>2>>>0],(x(),R)[o+12>>>2>>>0],(x(),R)[o+8>>>2>>>0],(x(),R)[o+4>>>2>>>0],(x(),R)[o>>>2>>>0],0),h=(x(),R)[o+32>>>2>>>0],y=l.getTimezoneOffset(),m=new Date(l.getFullYear(),6,1).getTimezoneOffset(),k=new Date(l.getFullYear(),0,1).getTimezoneOffset(),z=Math.min(k,m);return 0>h?(x(),R)[o+32>>>2>>>0]=+(m!=k&&z==y):0<h!=(z==y)&&(m=Math.max(k,m),l.setTime(l.getTime()+6e4*((0<h?z:m)-y))),(x(),R)[o+24>>>2>>>0]=l.getDay(),h=(Un(l.getFullYear())?Pn:qn)[l.getMonth()]+l.getDate()-1|0,(x(),R)[o+28>>>2>>>0]=h,(x(),R)[o>>>2>>>0]=l.getSeconds(),(x(),R)[o+4>>>2>>>0]=l.getMinutes(),(x(),R)[o+8>>>2>>>0]=l.getHours(),(x(),R)[o+12>>>2>>>0]=l.getDate(),(x(),R)[o+16>>>2>>>0]=l.getMonth(),(x(),R)[o+20>>>2>>>0]=l.getYear(),o=l.getTime(),BigInt(isNaN(o)?-1:o/1e3)}function Wn(o,l,h,y,m,k,z){return a?we(16,1,o,l,h,y,m,k,z):-52}function Ln(o,l,h,y,m,k){if(a)return we(17,1,o,l,h,y,m,k)}var Ht={},$m=()=>performance.timeOrigin+performance.now();function Vn(o,l){if(a)return we(18,1,o,l);if(Ht[o]&&(clearTimeout(Ht[o].id),delete Ht[o]),!l)return 0;var h=setTimeout(()=>{delete Ht[o],ti(()=>as(o,performance.timeOrigin+performance.now()))},l);return Ht[o]={id:h,ce:l},0}function vm(o,l,h,y){o>>>=0,l>>>=0,h>>>=0,y>>>=0;var m=new Date().getFullYear(),k=new Date(m,0,1).getTimezoneOffset();m=new Date(m,6,1).getTimezoneOffset();var z=Math.max(k,m);(x(),q)[o>>>2>>>0]=60*z,(x(),R)[l>>>2>>>0]=+(k!=m),o=(l=N=>{var W=Math.abs(N);return`UTC${0<=N?"-":"+"}${String(Math.floor(W/60)).padStart(2,"0")}${String(W%60).padStart(2,"0")}`})(k),l=l(m),m<k?(lt(o,h,17),lt(l,y,17)):(lt(o,y,17),lt(l,h,17))}var xm=()=>Date.now();function Sm(o,l,h){return h>>>=0,0<=o&&3>=o?(o===0?o=Date.now():o=performance.timeOrigin+performance.now(),o=Math.round(1e6*o),(x(),D)[h>>>3>>>0]=BigInt(o),0):28}var si=[],Gn=(o,l)=>{si.length=0;for(var h;h=(x(),te)[o++>>>0];){var y=h!=105;l+=(y&=h!=112)&&l%8?4:0,si.push(h==112?(x(),q)[l>>>2>>>0]:h==106?(x(),D)[l>>>3>>>0]:h==105?(x(),R)[l>>>2>>>0]:(x(),_e)[l>>>3>>>0]),l+=y?8:4}return si};function km(o,l,h){return o>>>=0,l=Gn(l>>>0,h>>>0),li[o](...l)}function Tm(o,l,h){return o>>>=0,l=Gn(l>>>0,h>>>0),li[o](...l)}var Im=()=>{};function Em(o,l){return C(Se(o>>>0,l>>>0))}var zm=()=>{throw Ze+=1,"unwind"};function Cm(){return 4294901760}var Am=()=>navigator.hardwareConcurrency,$t={};function $r(o){if(!(2147483648&(o>>>=0)))return Te("Cannot use emscripten_pc_get_function on native functions without -sUSE_OFFSET_CONVERTER"),0;if(!(o=$t[o]))return 0;var l;if(l=/^\s+at (.*) \(.*\)$/.exec(o))o=l[1];else{if(!(l=/^(.+?)@/.exec(o)))return 0;o=l[1]}Xe($r.ud??0),l=mr(o)+1;var h=Ft(l);return h&&lt(o,h,l),$r.ud=h,$r.ud}function Om(o){o>>>=0;var l=(x(),te).length;if(o<=l||4294901760<o)return!1;for(var h=1;4>=h;h*=2){var y=l*(1+.2/h);y=Math.min(y,o+100663296);e:{y=(Math.min(4294901760,65536*Math.ceil(Math.max(o,y)/65536))-G.buffer.byteLength+65535)/65536|0;try{G.grow(y),re();var m=1;break e}catch{}m=void 0}if(m)return!0}return!1}var vr=o=>{var l;if(l=/\bwasm-function\[\d+\]:(0x[0-9a-f]+)/.exec(o))return+l[1];if(/\bwasm-function\[(\d+)\]:(\d+)/.exec(o))Te("Legacy backtrace format detected but -sUSE_OFFSET_CONVERTER not present.");else if(l=/:(\d+):\d+(?:\)|$)/.exec(o))return 2147483648|+l[1];return 0},Hn=o=>{o.forEach(l=>{var h=vr(l);h&&($t[h]=l)})};function Rm(){var o=Error().stack.toString().split(`
`);return o[0]=="Error"&&o.shift(),Hn(o),$t.td=vr(o[3]),$t.Md=o,$t.td}function Bm(o,l,h){if(o>>>=0,l>>>=0,$t.td==o)var y=$t.Md;else(y=Error().stack.toString().split(`
`))[0]=="Error"&&y.shift(),Hn(y);for(var m=3;y[m]&&vr(y[m])!=o;)++m;for(o=0;o<h&&y[o+m];++o)(x(),R)[l+4*o>>>2>>>0]=vr(y[o+m]);return o}var oi,ui={},Fn=()=>{if(!oi){var o,l={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8",_:"./this.program"};for(o in ui)ui[o]===void 0?delete l[o]:l[o]=ui[o];var h=[];for(o in l)h.push(`${o}=${l[o]}`);oi=h}return oi};function jn(o,l){if(a)return we(19,1,o,l);o>>>=0,l>>>=0;var h,y=0,m=0;for(h of Fn()){var k=l+y;(x(),q)[o+m>>>2>>>0]=k,y+=lt(h,k,1/0)+1,m+=4}return 0}function Kn(o,l){if(a)return we(20,1,o,l);o>>>=0,l>>>=0;var h=Fn();for(var y of((x(),q)[o>>>2>>>0]=h.length,o=0,h))o+=mr(y)+1;return(x(),q)[l>>>2>>>0]=o,0}function Zn(o){return a?we(21,1,o):52}function Qn(o,l,h,y){return a?we(22,1,o,l,h,y):52}function Yn(o,l,h,y){return a?we(23,1,o,l,h,y):70}var Nm=[null,[],[]];function Xn(o,l,h,y){if(a)return we(24,1,o,l,h,y);l>>>=0,h>>>=0,y>>>=0;for(var m=0,k=0;k<h;k++){var z=(x(),q)[l>>>2>>>0],N=(x(),q)[l+4>>>2>>>0];l+=8;for(var W=0;W<N;W++){var H=o,ne=(x(),te)[z+W>>>0],de=Nm[H];ne===0||ne===10?((H===1?A:C)(gn(de)),de.length=0):de.push(ne)}m+=N}return(x(),q)[y>>>2>>>0]=m,0}function Dm(o){return o>>>0}a||function(){for(var o=t.numThreads-1;o--;)ln();Re.push(()=>{st++,function(l){a?l():Promise.all(ut.map(un)).then(l)}(()=>Pe())})}(),a||(G=new WebAssembly.Memory({initial:256,maximum:65536,shared:!0}),re()),t.wasmBinary&&(_=t.wasmBinary),t.stackSave=()=>oe(),t.stackRestore=o=>se(o),t.stackAlloc=o=>hi(o),t.setValue=function(o,l,h="i8"){switch(h.endsWith("*")&&(h="*"),h){case"i1":case"i8":(x(),V)[o>>>0]=l;break;case"i16":(x(),ee)[o>>>1>>>0]=l;break;case"i32":(x(),R)[o>>>2>>>0]=l;break;case"i64":(x(),D)[o>>>3>>>0]=BigInt(l);break;case"float":(x(),X)[o>>>2>>>0]=l;break;case"double":(x(),_e)[o>>>3>>>0]=l;break;case"*":(x(),q)[o>>>2>>>0]=l;break;default:Te(`invalid type for setValue: ${h}`)}},t.getValue=function(o,l="i8"){switch(l.endsWith("*")&&(l="*"),l){case"i1":case"i8":return(x(),V)[o>>>0];case"i16":return(x(),ee)[o>>>1>>>0];case"i32":return(x(),R)[o>>>2>>>0];case"i64":return(x(),D)[o>>>3>>>0];case"float":return(x(),X)[o>>>2>>>0];case"double":return(x(),_e)[o>>>3>>>0];case"*":return(x(),q)[o>>>2>>>0];default:Te(`invalid type for getValue: ${l}`)}},t.UTF8ToString=Se,t.stringToUTF8=lt,t.lengthBytesUTF8=mr;var Mm=[jr,an,cn,yn,_n,wn,bn,$n,vn,xn,Sn,kn,Tn,In,En,zn,Wn,Ln,Vn,jn,Kn,Zn,Qn,Yn,Xn],li={891356:(o,l,h,y,m)=>{if(t===void 0||!t.Zc)return 1;if((o=Se(Number(o>>>0))).startsWith("./")&&(o=o.substring(2)),!(o=t.Zc.get(o)))return 2;if(l=Number(l>>>0),h=Number(h>>>0),y=Number(y>>>0),l+h>o.byteLength)return 3;try{let k=o.subarray(l,l+h);switch(m){case 0:(x(),te).set(k,y>>>0);break;case 1:t.Yd?t.Yd(y,k):t.Ld(y,k);break;default:return 4}return 0}catch{return 4}},892180:(o,l,h)=>{t.xd(o,(x(),te).subarray(l>>>0,l+h>>>0))},892244:()=>t.$d(),892286:o=>{t.wd(o)},892323:()=>{t.Ed()},892354:()=>{t.Fd()},892383:()=>{t.Jd()},892408:o=>t.Dd(o),892441:o=>t.Hd(o),892473:(o,l,h)=>{t.jd(Number(o),Number(l),Number(h),!0)},892536:(o,l,h)=>{t.jd(Number(o),Number(l),Number(h))},892593:()=>typeof wasmOffsetConverter<"u",892650:o=>{t.ac("Abs",o,void 0)},892701:o=>{t.ac("Neg",o,void 0)},892752:o=>{t.ac("Floor",o,void 0)},892805:o=>{t.ac("Ceil",o,void 0)},892857:o=>{t.ac("Reciprocal",o,void 0)},892915:o=>{t.ac("Sqrt",o,void 0)},892967:o=>{t.ac("Exp",o,void 0)},893018:o=>{t.ac("Erf",o,void 0)},893069:o=>{t.ac("Sigmoid",o,void 0)},893124:(o,l,h)=>{t.ac("HardSigmoid",o,{alpha:l,beta:h})},893203:o=>{t.ac("Log",o,void 0)},893254:o=>{t.ac("Sin",o,void 0)},893305:o=>{t.ac("Cos",o,void 0)},893356:o=>{t.ac("Tan",o,void 0)},893407:o=>{t.ac("Asin",o,void 0)},893459:o=>{t.ac("Acos",o,void 0)},893511:o=>{t.ac("Atan",o,void 0)},893563:o=>{t.ac("Sinh",o,void 0)},893615:o=>{t.ac("Cosh",o,void 0)},893667:o=>{t.ac("Asinh",o,void 0)},893720:o=>{t.ac("Acosh",o,void 0)},893773:o=>{t.ac("Atanh",o,void 0)},893826:o=>{t.ac("Tanh",o,void 0)},893878:o=>{t.ac("Not",o,void 0)},893929:(o,l,h)=>{t.ac("Clip",o,{min:l,max:h})},893998:o=>{t.ac("Clip",o,void 0)},894050:(o,l)=>{t.ac("Elu",o,{alpha:l})},894108:o=>{t.ac("Gelu",o,void 0)},894160:o=>{t.ac("Relu",o,void 0)},894212:(o,l)=>{t.ac("LeakyRelu",o,{alpha:l})},894276:(o,l)=>{t.ac("ThresholdedRelu",o,{alpha:l})},894346:(o,l)=>{t.ac("Cast",o,{to:l})},894404:o=>{t.ac("Add",o,void 0)},894455:o=>{t.ac("Sub",o,void 0)},894506:o=>{t.ac("Mul",o,void 0)},894557:o=>{t.ac("Div",o,void 0)},894608:o=>{t.ac("Pow",o,void 0)},894659:o=>{t.ac("Equal",o,void 0)},894712:o=>{t.ac("Greater",o,void 0)},894767:o=>{t.ac("GreaterOrEqual",o,void 0)},894829:o=>{t.ac("Less",o,void 0)},894881:o=>{t.ac("LessOrEqual",o,void 0)},894940:(o,l,h,y,m)=>{t.ac("ReduceMean",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895115:(o,l,h,y,m)=>{t.ac("ReduceMax",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895289:(o,l,h,y,m)=>{t.ac("ReduceMin",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895463:(o,l,h,y,m)=>{t.ac("ReduceProd",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895638:(o,l,h,y,m)=>{t.ac("ReduceSum",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895812:(o,l,h,y,m)=>{t.ac("ReduceL1",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},895985:(o,l,h,y,m)=>{t.ac("ReduceL2",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},896158:(o,l,h,y,m)=>{t.ac("ReduceLogSum",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},896335:(o,l,h,y,m)=>{t.ac("ReduceSumSquare",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},896515:(o,l,h,y,m)=>{t.ac("ReduceLogSumExp",o,{keepDims:!!l,noopWithEmptyAxes:!!h,axes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},896695:o=>{t.ac("Where",o,void 0)},896748:(o,l,h)=>{t.ac("Transpose",o,{perm:l?Array.from((x(),R).subarray(Number(l)>>>0,Number(h)>>>0)):[]})},896872:(o,l,h,y)=>{t.ac("DepthToSpace",o,{blocksize:l,mode:Se(h),format:y?"NHWC":"NCHW"})},897005:(o,l,h,y)=>{t.ac("DepthToSpace",o,{blocksize:l,mode:Se(h),format:y?"NHWC":"NCHW"})},897138:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge,ct)=>{t.ac("ConvTranspose",o,{format:W?"NHWC":"NCHW",autoPad:l,dilations:[h],group:y,kernelShape:[m],pads:[k,z],strides:[N],wIsConst:()=>!!(x(),V)[H>>>0],outputPadding:ne?Array.from((x(),R).subarray(Number(ne)>>>0,Number(de)>>>0)):[],outputShape:fe?Array.from((x(),R).subarray(Number(fe)>>>0,Number(ge)>>>0)):[],activation:Se(ct)})},897571:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("ConvTranspose",o,{format:N?"NHWC":"NCHW",autoPad:l,dilations:Array.from((x(),R).subarray(Number(h)>>>0,2+(Number(h)>>>0)>>>0)),group:y,kernelShape:Array.from((x(),R).subarray(Number(m)>>>0,2+(Number(m)>>>0)>>>0)),pads:Array.from((x(),R).subarray(Number(k)>>>0,4+(Number(k)>>>0)>>>0)),strides:Array.from((x(),R).subarray(Number(z)>>>0,2+(Number(z)>>>0)>>>0)),wIsConst:()=>!!(x(),V)[W>>>0],outputPadding:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],outputShape:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[],activation:Se(ge)})},898232:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge,ct)=>{t.ac("ConvTranspose",o,{format:W?"NHWC":"NCHW",autoPad:l,dilations:[h],group:y,kernelShape:[m],pads:[k,z],strides:[N],wIsConst:()=>!!(x(),V)[H>>>0],outputPadding:ne?Array.from((x(),R).subarray(Number(ne)>>>0,Number(de)>>>0)):[],outputShape:fe?Array.from((x(),R).subarray(Number(fe)>>>0,Number(ge)>>>0)):[],activation:Se(ct)})},898665:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("ConvTranspose",o,{format:N?"NHWC":"NCHW",autoPad:l,dilations:Array.from((x(),R).subarray(Number(h)>>>0,2+(Number(h)>>>0)>>>0)),group:y,kernelShape:Array.from((x(),R).subarray(Number(m)>>>0,2+(Number(m)>>>0)>>>0)),pads:Array.from((x(),R).subarray(Number(k)>>>0,4+(Number(k)>>>0)>>>0)),strides:Array.from((x(),R).subarray(Number(z)>>>0,2+(Number(z)>>>0)>>>0)),wIsConst:()=>!!(x(),V)[W>>>0],outputPadding:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],outputShape:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[],activation:Se(ge)})},899326:(o,l)=>{t.ac("GlobalAveragePool",o,{format:l?"NHWC":"NCHW"})},899417:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("AveragePool",o,{format:ge?"NHWC":"NCHW",auto_pad:l,ceil_mode:h,count_include_pad:y,storage_order:m,dilations:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[],kernel_shape:N?Array.from((x(),R).subarray(Number(N)>>>0,Number(W)>>>0)):[],pads:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],strides:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[]})},899896:(o,l)=>{t.ac("GlobalAveragePool",o,{format:l?"NHWC":"NCHW"})},899987:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("AveragePool",o,{format:ge?"NHWC":"NCHW",auto_pad:l,ceil_mode:h,count_include_pad:y,storage_order:m,dilations:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[],kernel_shape:N?Array.from((x(),R).subarray(Number(N)>>>0,Number(W)>>>0)):[],pads:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],strides:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[]})},900466:(o,l)=>{t.ac("GlobalMaxPool",o,{format:l?"NHWC":"NCHW"})},900553:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("MaxPool",o,{format:ge?"NHWC":"NCHW",auto_pad:l,ceil_mode:h,count_include_pad:y,storage_order:m,dilations:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[],kernel_shape:N?Array.from((x(),R).subarray(Number(N)>>>0,Number(W)>>>0)):[],pads:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],strides:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[]})},901028:(o,l)=>{t.ac("GlobalMaxPool",o,{format:l?"NHWC":"NCHW"})},901115:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge)=>{t.ac("MaxPool",o,{format:ge?"NHWC":"NCHW",auto_pad:l,ceil_mode:h,count_include_pad:y,storage_order:m,dilations:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[],kernel_shape:N?Array.from((x(),R).subarray(Number(N)>>>0,Number(W)>>>0)):[],pads:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],strides:de?Array.from((x(),R).subarray(Number(de)>>>0,Number(fe)>>>0)):[]})},901590:(o,l,h,y,m)=>{t.ac("Gemm",o,{alpha:l,beta:h,transA:y,transB:m})},901694:o=>{t.ac("MatMul",o,void 0)},901748:(o,l,h,y)=>{t.ac("ArgMax",o,{keepDims:!!l,selectLastIndex:!!h,axis:y})},901856:(o,l,h,y)=>{t.ac("ArgMin",o,{keepDims:!!l,selectLastIndex:!!h,axis:y})},901964:(o,l)=>{t.ac("Softmax",o,{axis:l})},902027:(o,l)=>{t.ac("Concat",o,{axis:l})},902087:(o,l,h,y,m)=>{t.ac("Split",o,{axis:l,numOutputs:h,splitSizes:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},902243:o=>{t.ac("Expand",o,void 0)},902297:(o,l)=>{t.ac("Gather",o,{axis:Number(l)})},902368:(o,l)=>{t.ac("GatherElements",o,{axis:Number(l)})},902447:(o,l)=>{t.ac("GatherND",o,{batch_dims:Number(l)})},902526:(o,l,h,y,m,k,z,N,W,H,ne)=>{t.ac("Resize",o,{antialias:l,axes:h?Array.from((x(),R).subarray(Number(h)>>>0,Number(y)>>>0)):[],coordinateTransformMode:Se(m),cubicCoeffA:k,excludeOutside:z,extrapolationValue:N,keepAspectRatioPolicy:Se(W),mode:Se(H),nearestMode:Se(ne)})},902888:(o,l,h,y,m,k,z)=>{t.ac("Slice",o,{starts:l?Array.from((x(),R).subarray(Number(l)>>>0,Number(h)>>>0)):[],ends:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[],axes:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[]})},903152:o=>{t.ac("Tile",o,void 0)},903204:(o,l,h)=>{t.ac("InstanceNormalization",o,{epsilon:l,format:h?"NHWC":"NCHW"})},903318:(o,l,h)=>{t.ac("InstanceNormalization",o,{epsilon:l,format:h?"NHWC":"NCHW"})},903432:o=>{t.ac("Range",o,void 0)},903485:(o,l)=>{t.ac("Einsum",o,{equation:Se(l)})},903566:(o,l,h,y,m)=>{t.ac("Pad",o,{mode:l,value:h,pads:y?Array.from((x(),R).subarray(Number(y)>>>0,Number(m)>>>0)):[]})},903709:(o,l,h,y,m,k)=>{t.ac("BatchNormalization",o,{epsilon:l,momentum:h,spatial:!!m,trainingMode:!!y,format:k?"NHWC":"NCHW"})},903878:(o,l,h,y,m,k)=>{t.ac("BatchNormalization",o,{epsilon:l,momentum:h,spatial:!!m,trainingMode:!!y,format:k?"NHWC":"NCHW"})},904047:(o,l,h)=>{t.ac("CumSum",o,{exclusive:Number(l),reverse:Number(h)})},904144:(o,l,h)=>{t.ac("DequantizeLinear",o,{axis:l,blockSize:h})},904234:(o,l,h,y,m)=>{t.ac("GridSample",o,{align_corners:l,mode:Se(h),padding_mode:Se(y),format:m?"NHWC":"NCHW"})},904404:(o,l,h,y,m)=>{t.ac("GridSample",o,{align_corners:l,mode:Se(h),padding_mode:Se(y),format:m?"NHWC":"NCHW"})},904574:(o,l)=>{t.ac("ScatterND",o,{reduction:Se(l)})},904659:(o,l,h,y,m,k,z,N,W)=>{t.ac("Attention",o,{numHeads:l,isUnidirectional:h,maskFilterValue:y,scale:m,doRotary:k,qkvHiddenSizes:z?Array.from((x(),R).subarray(Number(N)>>>0,Number(N)+z>>>0)):[],pastPresentShareBuffer:!!W})},904931:o=>{t.ac("BiasAdd",o,void 0)},904986:o=>{t.ac("BiasSplitGelu",o,void 0)},905047:o=>{t.ac("FastGelu",o,void 0)},905103:(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge,ct,mi)=>{t.ac("Conv",o,{format:de?"NHWC":"NCHW",auto_pad:l,dilations:h?Array.from((x(),R).subarray(Number(h)>>>0,Number(y)>>>0)):[],group:m,kernel_shape:k?Array.from((x(),R).subarray(Number(k)>>>0,Number(z)>>>0)):[],pads:N?Array.from((x(),R).subarray(Number(N)>>>0,Number(W)>>>0)):[],strides:H?Array.from((x(),R).subarray(Number(H)>>>0,Number(ne)>>>0)):[],w_is_const:()=>!!(x(),V)[Number(fe)>>>0],activation:Se(ge),activation_params:ct?Array.from((x(),X).subarray(Number(ct)>>>0,Number(mi)>>>0)):[]})},905687:o=>{t.ac("Gelu",o,void 0)},905739:(o,l,h,y,m,k,z,N,W)=>{t.ac("GroupQueryAttention",o,{numHeads:l,kvNumHeads:h,scale:y,softcap:m,doRotary:k,rotaryInterleaved:z,smoothSoftmax:N,localWindowSize:W})},905956:(o,l,h,y)=>{t.ac("LayerNormalization",o,{axis:l,epsilon:h,simplified:!!y})},906067:(o,l,h,y)=>{t.ac("LayerNormalization",o,{axis:l,epsilon:h,simplified:!!y})},906178:(o,l,h,y,m,k)=>{t.ac("MatMulNBits",o,{k:l,n:h,accuracyLevel:y,bits:m,blockSize:k})},906305:(o,l,h,y,m,k)=>{t.ac("MultiHeadAttention",o,{numHeads:l,isUnidirectional:h,maskFilterValue:y,scale:m,doRotary:k})},906464:(o,l)=>{t.ac("QuickGelu",o,{alpha:l})},906528:(o,l,h,y,m)=>{t.ac("RotaryEmbedding",o,{interleaved:!!l,numHeads:h,rotaryEmbeddingDim:y,scale:m})},906667:(o,l,h)=>{t.ac("SkipLayerNormalization",o,{epsilon:l,simplified:!!h})},906769:(o,l,h)=>{t.ac("SkipLayerNormalization",o,{epsilon:l,simplified:!!h})},906871:(o,l,h,y)=>{t.ac("GatherBlockQuantized",o,{gatherAxis:l,quantizeAxis:h,blockSize:y})},906992:o=>{t.Id(o)},907026:(o,l)=>t.Kd(Number(o),Number(l),t.$c.Nd,t.$c.errors)};function Um(o,l,h){return Dn(async()=>{await t.Gd(Number(o),Number(l),Number(h))})}function Pm(){return typeof wasmOffsetConverter<"u"}var Jn,es,di,Xe,Ft,pi,ts,rs,is,ci,as,ns,ue,jt,ss,se,hi,oe,os,us,ls,ds,ps,cs,hs,fi,fs,ms,gs,ys,_s,ws,bs,$s,vs,xs,Ss,ks,Ts,Is,Es,zs,Cs,As,Os,Rs,Bs,Ns,Ds,Ms,Us,Ps,qs,Ws,Ls,Vs,Gs,Hs,Fs,js,Ks,Zs,Qs,Ys,pt=await async function(){function o(y,m){var k=pt=y.exports;y={};for(let[z,N]of Object.entries(k))typeof N=="function"?(k=Xf(N),y[z]=k):y[z]=N;return pt=y,pt=function(){var z=pt,N=H=>ne=>H(ne)>>>0,W=H=>()=>H()>>>0;return(z=Object.assign({},z)).ub=N(z.ub),z.Yb=W(z.Yb),z._b=N(z._b),z.mc=N(z.mc),z.nc=W(z.nc),z.rc=N(z.rc),z}(),nn.push(pt.$b),w=m,Jn=(m=pt).ub,es=m.vb,t._OrtInit=m.wb,t._OrtGetLastError=m.xb,t._OrtCreateSessionOptions=m.yb,t._OrtAppendExecutionProvider=m.zb,t._OrtAddFreeDimensionOverride=m.Ab,t._OrtAddSessionConfigEntry=m.Bb,t._OrtReleaseSessionOptions=m.Cb,t._OrtCreateSession=m.Db,t._OrtReleaseSession=m.Eb,t._OrtGetInputOutputCount=m.Fb,t._OrtGetInputOutputMetadata=m.Gb,t._OrtFree=m.Hb,t._OrtCreateTensor=m.Ib,t._OrtGetTensorData=m.Jb,t._OrtReleaseTensor=m.Kb,t._OrtCreateRunOptions=m.Lb,t._OrtAddRunConfigEntry=m.Mb,t._OrtReleaseRunOptions=m.Nb,t._OrtCreateBinding=m.Ob,t._OrtBindInput=m.Pb,t._OrtBindOutput=m.Qb,t._OrtClearBoundOutputs=m.Rb,t._OrtReleaseBinding=m.Sb,t._OrtRunWithBinding=m.Tb,t._OrtRun=m.Ub,t._OrtEndProfiling=m.Vb,t._JsepOutput=m.Wb,t._JsepGetNodeName=m.Xb,di=m.Yb,t._free=Xe=m.Zb,t._malloc=Ft=m._b,pi=m.bc,ts=m.cc,rs=m.dc,is=m.ec,ci=m.fc,as=m.gc,ns=m.hc,ue=m.ic,jt=m.jc,ss=m.kc,se=m.lc,hi=m.mc,oe=m.nc,os=m.oc,us=m.pc,ls=m.qc,ds=m.rc,ps=m.sc,cs=m.tc,hs=m.uc,fi=m.vc,fs=m.wc,ms=m.xc,gs=m.yc,ys=m.zc,_s=m.Ac,ws=m.Bc,bs=m.Cc,$s=m.Dc,vs=m.Ec,xs=m.Fc,Ss=m.Gc,ks=m.Hc,Ts=m.Ic,Is=m.Jc,Es=m.Kc,zs=m.Lc,Cs=m.Mc,As=m.Nc,Os=m.Oc,Rs=m.Qc,Bs=m.Rc,Ns=m.Sc,Ds=m.bd,Ms=m.cd,Us=m.hd,Ps=m.kd,qs=m.ld,Ws=m.md,Ls=m.nd,Vs=m.od,Gs=m.pd,Hs=m.qd,Fs=m.rd,js=m.sd,Ks=m.Rd,Zs=m.Sd,Qs=m.Td,Ys=m.Ud,Pe(),pt}st++;var l,h=ot();return t.instantiateWasm?new Promise(y=>{t.instantiateWasm(h,(m,k)=>{y(o(m,k))})}):a?new Promise(y=>{$=m=>{var k=new WebAssembly.Instance(m,ot());y(o(k,m))}}):(Ke??(Ke=t.locateFile?t.locateFile?t.locateFile("ort-wasm-simd-threaded.jsep.wasm",g):g+"ort-wasm-simd-threaded.jsep.wasm":new URL(""+new URL("ort-wasm-simd-threaded.jsep-Bvhpdk4G.wasm",import.meta.url).href,import.meta.url).href),l=await async function(y){var m=Ke;if(!_&&typeof WebAssembly.instantiateStreaming=="function"&&!U(m))try{var k=fetch(m,{credentials:"same-origin"});return await WebAssembly.instantiateStreaming(k,y)}catch(z){C(`wasm streaming compile failed: ${z}`),C("falling back to ArrayBuffer instantiation")}return async function(z,N){try{var W=await async function(H){if(!_)try{var ne=await d(H);return new Uint8Array(ne)}catch{}if(H==Ke&&_)H=new Uint8Array(_);else{if(!p)throw"both async and sync fetching of the wasm failed";H=p(H)}return H}(z);return await WebAssembly.instantiate(W,N)}catch(H){C(`failed to asynchronously prepare wasm: ${H}`),Te(H)}}(m,y)}(h),o(l.instance,l.module))}();function qm(o,l,h,y){var m=oe();try{return cs(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function Wm(o,l,h){var y=oe();try{return hs(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;ue(1,0)}}function Lm(o,l,h){var y=oe();try{ps(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;ue(1,0)}}function Vm(o,l){var h=oe();try{return fi(o,l)}catch(y){if(se(h),y!==y+0)throw y;ue(1,0)}}function Gm(o,l){var h=oe();try{ms(o,l)}catch(y){if(se(h),y!==y+0)throw y;ue(1,0)}}function Hm(o){var l=oe();try{gs(o)}catch(h){if(se(l),h!==h+0)throw h;ue(1,0)}}function Fm(o,l,h,y,m,k,z){var N=oe();try{return fs(o,l,h,y,m,k,z)}catch(W){if(se(N),W!==W+0)throw W;ue(1,0)}}function jm(o,l,h,y){var m=oe();try{ys(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function Km(o,l,h,y,m){var k=oe();try{_s(o,l,h,y,m)}catch(z){if(se(k),z!==z+0)throw z;ue(1,0)}}function Zm(o,l,h,y,m,k){var z=oe();try{ws(o,l,h,y,m,k)}catch(N){if(se(z),N!==N+0)throw N;ue(1,0)}}function Qm(o,l,h,y,m,k,z){var N=oe();try{bs(o,l,h,y,m,k,z)}catch(W){if(se(N),W!==W+0)throw W;ue(1,0)}}function Ym(o,l,h,y,m,k,z,N){var W=oe();try{$s(o,l,h,y,m,k,z,N)}catch(H){if(se(W),H!==H+0)throw H;ue(1,0)}}function Xm(o,l,h,y){var m=oe();try{vs(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function Jm(o,l,h,y,m,k,z){var N=oe();try{xs(o,l,h,y,m,k,z)}catch(W){if(se(N),W!==W+0)throw W;ue(1,0)}}function eg(o,l,h,y,m){var k=oe();try{return Ss(o,l,h,y,m)}catch(z){if(se(k),z!==z+0)throw z;ue(1,0)}}function tg(o,l,h,y,m,k,z,N){var W=oe();try{ks(o,l,h,y,m,k,z,N)}catch(H){if(se(W),H!==H+0)throw H;ue(1,0)}}function rg(o,l,h,y,m,k,z,N,W,H,ne,de){var fe=oe();try{Ts(o,l,h,y,m,k,z,N,W,H,ne,de)}catch(ge){if(se(fe),ge!==ge+0)throw ge;ue(1,0)}}function ig(o,l,h,y,m,k){var z=oe();try{return Is(o,l,h,y,m,k)}catch(N){if(se(z),N!==N+0)throw N;ue(1,0)}}function ag(o,l){var h=oe();try{return Es(o,l)}catch(y){if(se(h),y!==y+0)throw y;return ue(1,0),0n}}function ng(o,l,h,y,m,k,z,N,W){var H=oe();try{zs(o,l,h,y,m,k,z,N,W)}catch(ne){if(se(H),ne!==ne+0)throw ne;ue(1,0)}}function sg(o){var l=oe();try{return Cs(o)}catch(h){if(se(l),h!==h+0)throw h;ue(1,0)}}function og(o){var l=oe();try{return Os(o)}catch(h){if(se(l),h!==h+0)throw h;return ue(1,0),0n}}function ug(o,l,h,y,m,k){var z=oe();try{return Ps(o,l,h,y,m,k)}catch(N){if(se(z),N!==N+0)throw N;ue(1,0)}}function lg(o,l,h,y,m,k){var z=oe();try{return qs(o,l,h,y,m,k)}catch(N){if(se(z),N!==N+0)throw N;ue(1,0)}}function dg(o,l,h){var y=oe();try{return Ws(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;ue(1,0)}}function pg(o,l,h,y,m,k,z,N){var W=oe();try{return As(o,l,h,y,m,k,z,N)}catch(H){if(se(W),H!==H+0)throw H;ue(1,0)}}function cg(o,l,h,y,m){var k=oe();try{return Ls(o,l,h,y,m)}catch(z){if(se(k),z!==z+0)throw z;return ue(1,0),0n}}function hg(o,l,h,y){var m=oe();try{return Vs(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function fg(o,l,h,y){var m=oe();try{return Gs(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function mg(o,l,h,y,m,k,z,N,W,H,ne,de){var fe=oe();try{return Hs(o,l,h,y,m,k,z,N,W,H,ne,de)}catch(ge){if(se(fe),ge!==ge+0)throw ge;ue(1,0)}}function gg(o,l,h,y,m,k,z,N,W,H,ne){var de=oe();try{Ms(o,l,h,y,m,k,z,N,W,H,ne)}catch(fe){if(se(de),fe!==fe+0)throw fe;ue(1,0)}}function yg(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge,ct,mi){var Sg=oe();try{Us(o,l,h,y,m,k,z,N,W,H,ne,de,fe,ge,ct,mi)}catch(gi){if(se(Sg),gi!==gi+0)throw gi;ue(1,0)}}function _g(o,l,h,y){var m=oe();try{return Fs(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}function wg(o,l,h,y,m){var k=oe();try{return js(o,l,h,y,m)}catch(z){if(se(k),z!==z+0)throw z;ue(1,0)}}function bg(o,l,h){var y=oe();try{return Bs(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;return ue(1,0),0n}}function $g(o,l,h){var y=oe();try{return Rs(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;ue(1,0)}}function vg(o,l,h){var y=oe();try{return Ns(o,l,h)}catch(m){if(se(y),m!==m+0)throw m;ue(1,0)}}function xg(o,l,h,y){var m=oe();try{Ds(o,l,h,y)}catch(k){if(se(m),k!==k+0)throw k;ue(1,0)}}return function o(){if(0<st)_t=o;else if(a)S==null||S(t),ze();else{for(;0<Re.length;)Re.shift()(t);0<st?_t=o:(t.calledRun=!0,O||(ze(),S==null||S(t)))}}(),t.PTR_SIZE=4,K?t:new Promise((o,l)=>{S=o,v=l})}var rp,ro,jg=P(()=>{var e,t;rp=to,ro=(t=(e=globalThis.self)==null?void 0:e.name)==null?void 0:t.startsWith("em-pthread"),ro&&to()}),vi,ha,io,Be,ip,Sr,ao,no,xi,so,Si,ap,ki,np,Oa=P(()=>{Aa(),vi=typeof location>"u"?void 0:location.origin,ha=import.meta.url>"file:"&&import.meta.url<"file;",io=()=>{{if(ha){let e=URL;return new URL(new e("ort.bundle.min.mjs",import.meta.url).href,vi).href}return import.meta.url}},Be=io(),ip=()=>{if(Be&&!Be.startsWith("blob:"))return Be.substring(0,Be.lastIndexOf("/")+1)},Sr=(e,t)=>{try{let r=t??Be;return(r?new URL(e,r):new URL(e)).origin===vi}catch{return!1}},ao=(e,t)=>{let r=t??Be;try{return(r?new URL(e,r):new URL(e)).href}catch{return}},no=(e,t)=>`${t??"./"}${e}`,xi=async e=>{let t=await(await fetch(e,{credentials:"same-origin"})).blob();return URL.createObjectURL(t)},so=async e=>(await import(e)).default,Si=(Fg(),dr(Jd)).default,ap=async()=>{if(!Be)throw new Error("Failed to load proxy worker: cannot determine the script source URL.");if(Sr(Be))return[void 0,Si()];let e=await xi(Be);return[e,Si(e)]},ki=(jg(),dr(tp)).default,np=async(e,t,r,i)=>{let a=ki&&!(e||t);if(a)if(Be)a=Sr(Be);else if(i&&!r)a=!0;else throw new Error("cannot determine the script source URL.");if(a)return[void 0,ki];{let n="ort-wasm-simd-threaded.jsep.mjs",s=e??ao(n,t),u=r&&s&&!Sr(s,t),d=u?await xi(s):s??no(n,t);return[u?d:void 0,await so(d)]}}}),Ti,kr,Zt,Ii,oo,uo,lo,Ra,me,Bt=P(()=>{Oa(),kr=!1,Zt=!1,Ii=!1,oo=()=>{if(typeof SharedArrayBuffer>"u")return!1;try{return typeof MessageChannel<"u"&&new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]))}catch{return!1}},uo=()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,30,1,28,0,65,0,253,15,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,186,1,26,11]))}catch{return!1}},lo=()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,19,1,17,0,65,1,253,15,65,2,253,15,65,3,253,15,253,147,2,11]))}catch{return!1}},Ra=async e=>{if(kr)return Promise.resolve();if(Zt)throw new Error("multiple calls to 'initializeWebAssembly()' detected.");if(Ii)throw new Error("previous call to 'initializeWebAssembly()' failed.");Zt=!0;let t=e.initTimeout,r=e.numThreads;if(e.simd!==!1){if(e.simd==="relaxed"){if(!lo())throw new Error("Relaxed WebAssembly SIMD is not supported in the current environment.")}else if(!uo())throw new Error("WebAssembly SIMD is not supported in the current environment.")}let i=oo();r>1&&!i&&(typeof self<"u"&&!self.crossOriginIsolated&&console.warn("env.wasm.numThreads is set to "+r+", but this will not work unless you enable crossOriginIsolated mode. See https://web.dev/cross-origin-isolation-guide/ for more info."),console.warn("WebAssembly multi-threading is not supported in the current environment. Falling back to single-threading."),e.numThreads=r=1);let a=e.wasmPaths,n=typeof a=="string"?a:void 0,s=a==null?void 0:a.mjs,u=(s==null?void 0:s.href)??s,d=a==null?void 0:a.wasm,p=(d==null?void 0:d.href)??d,c=e.wasmBinary,[f,g]=await np(u,n,r>1,!!c||!!p),_=!1,w=[];if(t>0&&w.push(new Promise(b=>{setTimeout(()=>{_=!0,b()},t)})),w.push(new Promise((b,S)=>{let v={numThreads:r};if(c)v.wasmBinary=c;else if(p||n)v.locateFile=$=>p??n+$;else if(u&&u.indexOf("blob:")!==0)v.locateFile=$=>new URL($,u).href;else if(f){let $=ip();$&&(v.locateFile=I=>$+I)}g(v).then($=>{Zt=!1,kr=!0,Ti=$,b(),f&&URL.revokeObjectURL(f)},$=>{Zt=!1,Ii=!0,S($)})})),await Promise.race(w),_)throw new Error(`WebAssembly backend initializing failed due to timeout: ${t}ms`)},me=()=>{if(kr&&Ti)return Ti;throw new Error("WebAssembly is not initialized yet.")}}),Fe,Pr,he,Ba=P(()=>{Bt(),Fe=(e,t)=>{let r=me(),i=r.lengthBytesUTF8(e)+1,a=r._malloc(i);return r.stringToUTF8(e,a,i),t.push(a),a},Pr=(e,t,r,i)=>{if(typeof e=="object"&&e!==null){if(r.has(e))throw new Error("Circular reference in options");r.add(e)}Object.entries(e).forEach(([a,n])=>{let s=t?t+a:a;if(typeof n=="object")Pr(n,s+".",r,i);else if(typeof n=="string"||typeof n=="number")i(s,n.toString());else if(typeof n=="boolean")i(s,n?"1":"0");else throw new Error(`Can't handle extra config type: ${typeof n}`)})},he=e=>{let t=me(),r=t.stackSave();try{let i=t.PTR_SIZE,a=t.stackAlloc(2*i);t._OrtGetLastError(a,a+i);let n=Number(t.getValue(a,i===4?"i32":"i64")),s=t.getValue(a+i,"*"),u=s?t.UTF8ToString(s):"";throw new Error(`${e} ERROR_CODE: ${n}, ERROR_MESSAGE: ${u}`)}finally{t.stackRestore(r)}}}),sp,Kg=P(()=>{Bt(),Ba(),sp=e=>{let t=me(),r=0,i=[],a=e||{};try{if((e==null?void 0:e.logSeverityLevel)===void 0)a.logSeverityLevel=2;else if(typeof e.logSeverityLevel!="number"||!Number.isInteger(e.logSeverityLevel)||e.logSeverityLevel<0||e.logSeverityLevel>4)throw new Error(`log severity level is not valid: ${e.logSeverityLevel}`);if((e==null?void 0:e.logVerbosityLevel)===void 0)a.logVerbosityLevel=0;else if(typeof e.logVerbosityLevel!="number"||!Number.isInteger(e.logVerbosityLevel))throw new Error(`log verbosity level is not valid: ${e.logVerbosityLevel}`);(e==null?void 0:e.terminate)===void 0&&(a.terminate=!1);let n=0;return(e==null?void 0:e.tag)!==void 0&&(n=Fe(e.tag,i)),r=t._OrtCreateRunOptions(a.logSeverityLevel,a.logVerbosityLevel,!!a.terminate,n),r===0&&he("Can't create run options."),(e==null?void 0:e.extra)!==void 0&&Pr(e.extra,"",new WeakSet,(s,u)=>{let d=Fe(s,i),p=Fe(u,i);t._OrtAddRunConfigEntry(r,d,p)!==0&&he(`Can't set a run config entry: ${s} - ${u}.`)}),[r,i]}catch(n){throw r!==0&&t._OrtReleaseRunOptions(r),i.forEach(s=>t._free(s)),n}}}),po,co,ho,Qt,fo,op,Zg=P(()=>{Bt(),Ba(),po=e=>{switch(e){case"disabled":return 0;case"basic":return 1;case"extended":return 2;case"layout":return 3;case"all":return 99;default:throw new Error(`unsupported graph optimization level: ${e}`)}},co=e=>{switch(e){case"sequential":return 0;case"parallel":return 1;default:throw new Error(`unsupported execution mode: ${e}`)}},ho=e=>{e.extra||(e.extra={}),e.extra.session||(e.extra.session={});let t=e.extra.session;t.use_ort_model_bytes_directly||(t.use_ort_model_bytes_directly="1"),e.executionProviders&&e.executionProviders.some(r=>(typeof r=="string"?r:r.name)==="webgpu")&&(e.enableMemPattern=!1)},Qt=(e,t,r,i)=>{let a=Fe(t,i),n=Fe(r,i);me()._OrtAddSessionConfigEntry(e,a,n)!==0&&he(`Can't set a session config entry: ${t} - ${r}.`)},fo=async(e,t,r)=>{for(let i of t){let a=typeof i=="string"?i:i.name,n=[];switch(a){case"webnn":if(a="WEBNN",typeof i!="string"){let c=i==null?void 0:i.deviceType;c&&Qt(e,"deviceType",c,r)}break;case"webgpu":if(a="JS",typeof i!="string"){let c=i;if(c!=null&&c.preferredLayout){if(c.preferredLayout!=="NCHW"&&c.preferredLayout!=="NHWC")throw new Error(`preferredLayout must be either 'NCHW' or 'NHWC': ${c.preferredLayout}`);Qt(e,"preferredLayout",c.preferredLayout,r)}}break;case"wasm":case"cpu":continue;default:throw new Error(`not supported execution provider: ${a}`)}let s=Fe(a,r),u=n.length,d=0,p=0;if(u>0){d=me()._malloc(u*me().PTR_SIZE),r.push(d),p=me()._malloc(u*me().PTR_SIZE),r.push(p);for(let c=0;c<u;c++)me().setValue(d+c*me().PTR_SIZE,n[c][0],"*"),me().setValue(p+c*me().PTR_SIZE,n[c][1],"*")}await me()._OrtAppendExecutionProvider(e,s,d,p,u)!==0&&he(`Can't append execution provider: ${a}.`)}},op=async e=>{let t=me(),r=0,i=[],a=e||{};ho(a);try{let n=po(a.graphOptimizationLevel??"all"),s=co(a.executionMode??"sequential"),u=typeof a.logId=="string"?Fe(a.logId,i):0,d=a.logSeverityLevel??2;if(!Number.isInteger(d)||d<0||d>4)throw new Error(`log severity level is not valid: ${d}`);let p=a.logVerbosityLevel??0;if(!Number.isInteger(p)||p<0||p>4)throw new Error(`log verbosity level is not valid: ${p}`);let c=typeof a.optimizedModelFilePath=="string"?Fe(a.optimizedModelFilePath,i):0;if(r=t._OrtCreateSessionOptions(n,!!a.enableCpuMemArena,!!a.enableMemPattern,s,!!a.enableProfiling,0,u,d,p,c),r===0&&he("Can't create session options."),a.executionProviders&&await fo(r,a.executionProviders,i),a.enableGraphCapture!==void 0){if(typeof a.enableGraphCapture!="boolean")throw new Error(`enableGraphCapture must be a boolean value: ${a.enableGraphCapture}`);Qt(r,"enableGraphCapture",a.enableGraphCapture.toString(),i)}if(a.freeDimensionOverrides)for(let[f,g]of Object.entries(a.freeDimensionOverrides)){if(typeof f!="string")throw new Error(`free dimension override name must be a string: ${f}`);if(typeof g!="number"||!Number.isInteger(g)||g<0)throw new Error(`free dimension override value must be a non-negative integer: ${g}`);let _=Fe(f,i);t._OrtAddFreeDimensionOverride(r,_,g)!==0&&he(`Can't set a free dimension override: ${f} - ${g}.`)}return a.extra!==void 0&&Pr(a.extra,"",new WeakSet,(f,g)=>{Qt(r,f,g,i)}),[r,i]}catch(n){throw r!==0&&t._OrtReleaseSessionOptions(r)!==0&&he("Can't release session options."),i.forEach(s=>t._free(s)),n}}}),It,at,Et,Fr,qr,Na,Da,fa,J=P(()=>{It=e=>{switch(e){case"int8":return 3;case"uint8":return 2;case"bool":return 9;case"int16":return 5;case"uint16":return 4;case"int32":return 6;case"uint32":return 12;case"float16":return 10;case"float32":return 1;case"float64":return 11;case"string":return 8;case"int64":return 7;case"uint64":return 13;case"int4":return 22;case"uint4":return 21;default:throw new Error(`unsupported data type: ${e}`)}},at=e=>{switch(e){case 3:return"int8";case 2:return"uint8";case 9:return"bool";case 5:return"int16";case 4:return"uint16";case 6:return"int32";case 12:return"uint32";case 10:return"float16";case 1:return"float32";case 11:return"float64";case 8:return"string";case 7:return"int64";case 13:return"uint64";case 22:return"int4";case 21:return"uint4";default:throw new Error(`unsupported data type: ${e}`)}},Et=(e,t)=>{let r=[-1,4,1,1,2,2,4,8,-1,1,2,8,4,8,-1,-1,-1,-1,-1,-1,-1,.5,.5][e],i=typeof t=="number"?t:t.reduce((a,n)=>a*n,1);return r>0?Math.ceil(i*r):void 0},Fr=e=>{switch(e){case"float16":return typeof Float16Array<"u"&&Float16Array.from?Float16Array:Uint16Array;case"float32":return Float32Array;case"uint8":return Uint8Array;case"int8":return Int8Array;case"uint16":return Uint16Array;case"int16":return Int16Array;case"int32":return Int32Array;case"bool":return Uint8Array;case"float64":return Float64Array;case"uint32":return Uint32Array;case"int64":return BigInt64Array;case"uint64":return BigUint64Array;default:throw new Error(`unsupported type: ${e}`)}},qr=e=>{switch(e){case"verbose":return 0;case"info":return 1;case"warning":return 2;case"error":return 3;case"fatal":return 4;default:throw new Error(`unsupported logging level: ${e}`)}},Na=e=>e==="float32"||e==="float16"||e==="int32"||e==="int64"||e==="uint32"||e==="uint8"||e==="bool"||e==="uint4"||e==="int4",Da=e=>e==="float32"||e==="float16"||e==="int32"||e==="int64"||e==="uint32"||e==="uint64"||e==="int8"||e==="uint8"||e==="bool"||e==="uint4"||e==="int4",fa=e=>{switch(e){case"none":return 0;case"cpu":return 1;case"cpu-pinned":return 2;case"texture":return 3;case"gpu-buffer":return 4;case"ml-tensor":return 5;default:throw new Error(`unsupported data location: ${e}`)}}}),Ma,up=P(()=>{Aa(),Ma=async e=>{if(typeof e=="string"){let t=await fetch(e);if(!t.ok)throw new Error(`failed to load external data file: ${e}`);let r=t.headers.get("Content-Length"),i=r?parseInt(r,10):0;if(i<1073741824)return new Uint8Array(await t.arrayBuffer());{if(!t.body)throw new Error(`failed to load external data file: ${e}, no response body.`);let a=t.body.getReader(),n;try{n=new ArrayBuffer(i)}catch(u){if(u instanceof RangeError){let d=Math.ceil(i/65536);n=new WebAssembly.Memory({initial:d,maximum:d}).buffer}else throw u}let s=0;for(;;){let{done:u,value:d}=await a.read();if(u)break;let p=d.byteLength;new Uint8Array(n,s,p).set(d),s+=p}return new Uint8Array(n,0,i)}}else return e instanceof Blob?new Uint8Array(await e.arrayBuffer()):e instanceof Uint8Array?e:new Uint8Array(e)}}),mo,go,yo,_o,Ua,wo,le,nt=P(()=>{J(),mo=["V","I","W","E","F"],go=(e,t)=>{console.log(`[${mo[e]},${new Date().toISOString()}]${t}`)},Ua=(e,t)=>{yo=e,_o=t},wo=(e,t)=>{let r=qr(e),i=qr(yo);r>=i&&go(r,typeof t=="function"?t():t)},le=(...e)=>{_o&&wo(...e)}}),bo,Wt,B,Wr,lp,dp,pp,ie=P(()=>{bo=class{static calcMatMulShape(e,t){return e[1]!==t[0]?void 0:[e[0],t[1]]}},Wt=class{static calcShape(e,t,r=!1){let i=e.length,a=t.length;if(i===0)return t;if(a===0)return e;let n=Math.max(e.length,t.length),s=new Array(n);if(r){if(i<2||a<2)return;let u=bo.calcMatMulShape([e[i-2],e[i-1]],[t[a-2],t[a-1]]);if(u===void 0)return;[s[n-2],s[n-1]]=u}for(let u=r?3:1;u<=n;u++){let d=i-u<0?1:e[i-u],p=a-u<0?1:t[a-u];if(d!==p&&d>1&&p>1)return;let c=Math.max(d,p);if(d&&p)s[n-u]=Math.max(d,p);else{if(c>1)return;s[n-u]=0}}return s}static isValidBroadcast(e,t){let r=e.length,i=t.length;if(r>i)return!1;for(let a=1;a<=r;a++)if(e[r-a]!==1&&e[r-a]!==t[i-a])return!1;return!0}},B=class Dr{static size(t){return Dr.getSizeFromDimensionRange(t,0,t.length)}static convertShape(t,r=4){let i=t.length;if(i===0)return[];let a=new Array(i),n=i-1;for(;n>=0;){if(t[n]%r===0){a[n]=t[n]/r;break}if(r%t[n]!==0)throw new Error("cannot convert shape");a[n]=1,r/=t[n],n--}for(n--;n>=0;n--)a[n]=t[n];return a}static sizeFromDimension(t,r){if(r<0||r>t.length)throw new Error(`invalid dimension of ${r} for sizeFromDimension as Tensor has ${t.length} dimensions.`);return Dr.getSizeFromDimensionRange(t,r,t.length)}static sizeToDimension(t,r){if(r<0||r>t.length)throw new Error(`invalid dimension of ${r} for sizeToDimension as Tensor has ${t.length} dimensions.`);return Dr.getSizeFromDimensionRange(t,0,r)}static getSizeFromDimensionRange(t,r,i){let a=1;for(let n=r;n<i;n++){if(t[n]<0)throw new Error("cannot get valid size from specified dimension range. Most likely the range contains negative values in them.");a*=Number(t[n])}return a}static computeStrides(t){let r=t.length;if(r===0)return[];if(r===1)return[1];let i=new Array(r);i[r-1]=1,i[r-2]=t[r-1];for(let a=r-3;a>=0;--a)i[a]=i[a+1]*t[a+1];return i}static normalizeAxis(t,r){if(t<-r&&t>=r)throw new Error("unsupported axis for this operation.");return t<0?t+r:t}static normalizeAxes(t,r){return t.map(i=>this.normalizeAxis(i,r??t.length))}static sortBasedOnPerm(t,r){return r?r.map(i=>t[i]):t.slice().reverse()}static padShape(t,r){let i=t.length;return t.map((a,n)=>a+r[n]+r[n+i])}static areEqual(t,r){return t.length!==r.length?!1:t.every((i,a)=>i===r[a])}},Wr=class sr{static adjustPoolAttributes(t,r,i,a,n,s){if(!t&&i.length!==r.length-2)throw new Error("length of specified kernel shapes should be 2 less than length of input dimensions");if(t)for(let u=0;u<r.length-2;u++)u>=i.length?i.push(r[u+2]):i[u]=r[u+2];for(let u=0;u<i.length;u++)if(u<a.length){if(a[u]<0)throw new Error("strides should be greater than or equal to 1")}else a.push(1);for(let u=0;u<i.length;u++)if(u<n.length){if(n[u]<0)throw new Error("dilations should be greater than or equal to 1")}else n.push(1);for(let u=0;u<i.length*2;u++)if(u<s.length){if(s[u]<0)throw new Error("pad should be greater than or equal to 1")}else s.push(0);for(let u=0;u<i.length;u++){if(i[u]<=0)throw new Error("kernel shapes need to be greater than 0");if(s[u]>=i[u]||s[u+i.length]>=i[u])throw new Error("pads should be smaller than kernel")}}static adjustPadsBasedOnAutoPad(t,r,i,a,n,s,u){if(u){if(n.length!==2*(t.length-2))throw new Error("length of pads should be twice the length of data dimensions");if(r.length!==t.length-2)throw new Error("length of strides should be the length of data dimensions");if(a.length!==t.length-2)throw new Error("length of kernel shapes should be the length of data dimensions");for(let d=0;d<t.length-2;d++)sr.adjustPadAndReturnShape(t[d+(s?1:2)],r[d],i[d],a[d],n,d,d+t.length-2,u)}}static computePoolOutputShape(t,r,i,a,n,s,u){if(r.length<=0)throw new Error("input shape must be of size greater than 0");let d=[r[0],r[1]];return sr.computeShapeHelper(t,r,d,i,a,n,s,u),d}static computeConvOutputShape(t,r,i,a,n,s,u){if(t.length<=0||r.length<=0)throw new Error("invalid input tensor dims or invalid filter tensor dims");let d=[t[0],r[0]];return sr.computeShapeHelper(!1,t,d,i,a,n,s,u),d}static computeShapeHelper(t,r,i,a,n,s,u,d){if(t)for(let p=0;p<r.length-2;p++)i.push(1);else for(let p=0;p<r.length-2;p++)i.push(sr.adjustPadAndReturnShape(r[p+2],a[p],n[p],s[p],u,p,p+r.length-2,d))}static adjustPadAndReturnShape(t,r,i,a,n,s,u,d){let p=i*(a-1)+1;if(d&&d!=="NOTSET")switch(d){case"VALID":return n[s]=0,n[u]=0,Math.floor((t-p)/r+1);case"SAME_LOWER":case"SAME_UPPER":if(i!==1)throw new Error("Dilation not supported for SAME_UPPER or SAME_LOWER");{let c=((t+r-1)/r-1)*r+a-t;return n[s]=Math.floor(d==="SAME_LOWER"?(c+1)/2:c/2),n[u]=c-n[s],Math.floor((t+c-a)/r+1)}default:throw new Error("Unsupported AutoPad type")}else return Math.floor((t+n[s]+n[u]-p)/r+1)}},lp=class{static getShapeOfGemmResult(e,t,r,i,a){if(e.length!==2||r.length!==2)throw new Error("shape need to be of size 2");let n,s,u;t?(n=e[1],s=e[0]):(n=e[0],s=e[1]);let d=-1;if(i?(u=r[0],d=1):(u=r[1],d=0),r[d]!==s)throw new Error("dimension mismatch");if(n<=0||u<=0||s<=0)throw new Error("invalid shape specified");if(a&&!Wt.isValidBroadcast(a,[n,u]))throw new Error("gemm: invalid bias shape for broadcast");return[n,u,s]}},dp=-34028234663852886e22,pp=34028234663852886e22}),Pa,cp=P(()=>{J(),Pa=(e,t)=>new(Fr(t))(e)}),Ei,ma,zi,$o,Ci,vo,Ai,Oi,Ri,xo,hp,Qg=P(()=>{J(),nt(),Ei=new Map([["float32",32],["float16",16],["int32",32],["uint32",32],["int64",64],["uint64",64],["int8",8],["uint8",8],["int4",4],["uint4",4]]),ma=(e,t)=>{if(t==="int32")return e;let r=Ei.get(t);if(!r)throw new Error(`WebNN backend does not support data type: ${t}`);let i=r/8;if(e.byteLength%i!==0)throw new Error(`Invalid Uint8Array length - must be a multiple of ${i}.`);let a=e.byteLength/i,n=new(Fr(t))(e.buffer,e.byteOffset,a);switch(t){case"int64":case"uint64":{let s=new Int32Array(a);for(let u=0;u<a;u++){let d=n[u];if(d>2147483647n||d<-2147483648n)throw new Error("Can not convert int64 data to int32 - value out of range.");s[u]=Number(d)}return new Uint8Array(s.buffer)}case"int8":case"uint8":case"uint32":{if(t==="uint32"&&n.some(u=>u>2147483647))throw new Error("Can not convert uint32 data to int32 - value out of range.");let s=Int32Array.from(n,Number);return new Uint8Array(s.buffer)}default:throw new Error(`Unsupported data conversion from ${t} to 'int32'`)}},zi=(e,t)=>{if(t==="int32")return e;if(e.byteLength%4!==0)throw new Error("Invalid Uint8Array length - must be a multiple of 4 (int32).");let r=e.byteLength/4,i=new Int32Array(e.buffer,e.byteOffset,r);switch(t){case"int64":{let a=BigInt64Array.from(i,BigInt);return new Uint8Array(a.buffer)}case"uint64":{if(i.some(n=>n<0))throw new Error("Can not convert int32 data to uin64 - negative value found.");let a=BigUint64Array.from(i,BigInt);return new Uint8Array(a.buffer)}case"int8":{if(i.some(n=>n<-128||n>127))throw new Error("Can not convert int32 data to int8 - value out of range.");let a=Int8Array.from(i,Number);return new Uint8Array(a.buffer)}case"uint8":{if(i.some(a=>a<0||a>255))throw new Error("Can not convert int32 data to uint8 - value out of range.");return Uint8Array.from(i,Number)}case"uint32":{if(i.some(n=>n<0))throw new Error("Can not convert int32 data to uint32 - negative value found.");let a=Uint32Array.from(i,Number);return new Uint8Array(a.buffer)}default:throw new Error(`Unsupported data conversion from 'int32' to ${t}`)}},$o=1,Ci=()=>$o++,vo=new Map([["int8","int32"],["uint8","int32"],["uint32","int32"],["int64","int32"]]),Ai=(e,t)=>{let r=Ei.get(e);if(!r)throw new Error(`WebNN backend does not support data type: ${e}`);return t.length>0?Math.ceil(t.reduce((i,a)=>i*a)*r/8):0},Oi=class{constructor(e){this.isDataConverted=!1;let{sessionId:t,context:r,tensor:i,dataType:a,shape:n,fallbackDataType:s}=e;this.sessionId=t,this.mlContext=r,this.mlTensor=i,this.dataType=a,this.tensorShape=n,this.fallbackDataType=s}get tensor(){return this.mlTensor}get type(){return this.dataType}get fallbackType(){return this.fallbackDataType}get shape(){return this.tensorShape}get byteLength(){return Ai(this.dataType,this.tensorShape)}destroy(){le("verbose",()=>"[WebNN] TensorWrapper.destroy"),this.mlTensor.destroy()}write(e){this.mlContext.writeTensor(this.mlTensor,e)}async read(e){if(this.fallbackDataType){let t=await this.mlContext.readTensor(this.mlTensor),r=zi(new Uint8Array(t),this.dataType);if(e){(e instanceof ArrayBuffer?new Uint8Array(e):new Uint8Array(e.buffer,e.byteOffset,e.byteLength)).set(r);return}else return r.buffer}else return e?this.mlContext.readTensor(this.mlTensor,e):this.mlContext.readTensor(this.mlTensor)}canReuseTensor(e,t,r){return this.mlContext===e&&this.dataType===t&&this.tensorShape.length===r.length&&this.tensorShape.every((i,a)=>i===r[a])}setIsDataConverted(e){this.isDataConverted=e}},Ri=class{constructor(e,t){this.tensorManager=e,this.wrapper=t}get tensorWrapper(){return this.wrapper}releaseTensor(){this.tensorWrapper&&(this.tensorManager.releaseTensor(this.tensorWrapper),this.wrapper=void 0)}async ensureTensor(e,t,r,i){let a=this.tensorManager.getMLContext(e),n;if(!a.opSupportLimits().input.dataTypes.includes(t)){if(n=vo.get(t),!n||!a.opSupportLimits().input.dataTypes.includes(n))throw new Error(`WebNN backend does not support data type: ${t}`);le("verbose",()=>`[WebNN] TensorIdTracker.ensureTensor: fallback dataType from ${t} to ${n}`)}if(this.wrapper){if(this.wrapper.canReuseTensor(a,t,r))return this.wrapper.tensor;if(i){if(this.wrapper.byteLength!==Ai(t,r))throw new Error("Unable to copy data to tensor with different size.");this.activeUpload=new Uint8Array(await this.wrapper.read())}this.tensorManager.releaseTensor(this.wrapper)}let s=typeof MLTensorUsage>"u"?void 0:MLTensorUsage.READ|MLTensorUsage.WRITE;return this.wrapper=await this.tensorManager.getCachedTensor(e,t,r,s,!0,!0,n),i&&this.activeUpload&&(this.wrapper.write(this.activeUpload),this.activeUpload=void 0),this.wrapper.tensor}upload(e){let t=e;if(this.wrapper){if(this.wrapper.fallbackType)if(this.wrapper.fallbackType==="int32")t=ma(e,this.wrapper.type),this.wrapper.setIsDataConverted(!0);else throw new Error(`Unsupported fallback data type: ${this.wrapper.fallbackType}`);if(e.byteLength===this.wrapper.byteLength){this.wrapper.write(t);return}else le("verbose",()=>"Data size does not match tensor size. Releasing tensor."),this.releaseTensor()}this.activeUpload?this.activeUpload.set(t):this.activeUpload=new Uint8Array(t)}async download(e){var t,r;if(this.activeUpload){let i=(t=this.wrapper)!=null&&t.isDataConverted?zi(this.activeUpload,(r=this.wrapper)==null?void 0:r.type):this.activeUpload;if(e){e instanceof ArrayBuffer?new Uint8Array(e).set(i):new Uint8Array(e.buffer,e.byteOffset,e.byteLength).set(i);return}else return i.buffer}if(!this.wrapper)throw new Error("Tensor has not been created.");return e?this.wrapper.read(e):this.wrapper.read()}},xo=class{constructor(e){this.backend=e,this.tensorTrackersById=new Map,this.freeTensors=[],this.externalTensors=new Set}getMLContext(e){let t=this.backend.getMLContext(e);if(!t)throw new Error("MLContext not found for session.");return t}reserveTensorId(){let e=Ci();return this.tensorTrackersById.set(e,new Ri(this)),e}releaseTensorId(e){let t=this.tensorTrackersById.get(e);t&&(this.tensorTrackersById.delete(e),t.tensorWrapper&&this.releaseTensor(t.tensorWrapper))}async ensureTensor(e,t,r,i,a){le("verbose",()=>`[WebNN] TensorManager.ensureTensor {tensorId: ${t}, dataType: ${r}, shape: ${i}, copyOld: ${a}}`);let n=this.tensorTrackersById.get(t);if(!n)throw new Error("Tensor not found.");return n.ensureTensor(e,r,i,a)}upload(e,t){let r=this.tensorTrackersById.get(e);if(!r)throw new Error("Tensor not found.");r.upload(t)}async download(e,t){le("verbose",()=>`[WebNN] TensorManager.download {tensorId: ${e}, dstBuffer: ${t==null?void 0:t.byteLength}}`);let r=this.tensorTrackersById.get(e);if(!r)throw new Error("Tensor not found.");return r.download(t)}releaseTensorsForSession(e){for(let t of this.freeTensors)t.sessionId===e&&t.destroy();this.freeTensors=this.freeTensors.filter(t=>t.sessionId!==e)}registerTensor(e,t,r,i){let a=this.getMLContext(e),n=Ci(),s=new Oi({sessionId:e,context:a,tensor:t,dataType:r,shape:i});return this.tensorTrackersById.set(n,new Ri(this,s)),this.externalTensors.add(s),n}async getCachedTensor(e,t,r,i,a,n,s){let u=this.getMLContext(e);for(let[p,c]of this.freeTensors.entries())if(c.canReuseTensor(u,t,r)){le("verbose",()=>`[WebNN] Reusing tensor {dataType: ${t}, ${s?`fallbackDataType: ${s},`:""} shape: ${r}`);let f=this.freeTensors.splice(p,1)[0];return f.sessionId=e,f}le("verbose",()=>`[WebNN] MLContext.createTensor {dataType: ${t}, ${s?`fallbackDataType: ${s},`:""} shape: ${r}}`);let d=await u.createTensor({dataType:s??t,shape:r,dimensions:r,usage:i,writable:a,readable:n});return new Oi({sessionId:e,context:u,tensor:d,dataType:t,shape:r,fallbackDataType:s})}releaseTensor(e){this.externalTensors.has(e)&&this.externalTensors.delete(e),this.freeTensors.push(e)}},hp=(...e)=>new xo(...e)}),Yt,So,fp,Yg=P(()=>{J(),Bt(),cp(),Qg(),nt(),Yt=new Map([[1,"float32"],[10,"float16"],[6,"int32"],[12,"uint32"],[7,"int64"],[13,"uint64"],[22,"int4"],[21,"uint4"],[3,"int8"],[2,"uint8"],[9,"uint8"]]),So=(e,t)=>{if(e===t)return!0;if(e===void 0||t===void 0)return!1;let r=Object.keys(e).sort(),i=Object.keys(t).sort();return r.length===i.length&&r.every((a,n)=>a===i[n]&&e[a]===t[a])},fp=class{constructor(e){this.tensorManager=hp(this),this.mlContextBySessionId=new Map,this.sessionIdsByMLContext=new Map,this.mlContextCache=[],this.sessionGraphInputs=new Map,this.sessionGraphOutputs=new Map,this.temporaryGraphInputs=[],this.temporaryGraphOutputs=[],this.temporarySessionTensorIds=new Map,Ua(e.logLevel,!!e.debug)}get currentSessionId(){if(this.activeSessionId===void 0)throw new Error("No active session");return this.activeSessionId}onRunStart(e){le("verbose",()=>`[WebNN] onRunStart {sessionId: ${e}}`),this.activeSessionId=e}onRunEnd(e){le("verbose",()=>`[WebNN] onRunEnd {sessionId: ${e}}`);let t=this.temporarySessionTensorIds.get(e);if(t){for(let r of t)le("verbose",()=>`[WebNN] releasing temporary tensor {tensorId: ${r}}`),this.tensorManager.releaseTensorId(r);this.temporarySessionTensorIds.delete(e),this.activeSessionId=void 0}}async createMLContext(e){if(e instanceof GPUDevice){let r=this.mlContextCache.findIndex(i=>i.gpuDevice===e);if(r!==-1)return this.mlContextCache[r].mlContext;{let i=await navigator.ml.createContext(e);return this.mlContextCache.push({gpuDevice:e,mlContext:i}),i}}else if(e===void 0){let r=this.mlContextCache.findIndex(i=>i.options===void 0&&i.gpuDevice===void 0);if(r!==-1)return this.mlContextCache[r].mlContext;{let i=await navigator.ml.createContext();return this.mlContextCache.push({mlContext:i}),i}}let t=this.mlContextCache.findIndex(r=>So(r.options,e));if(t!==-1)return this.mlContextCache[t].mlContext;{let r=await navigator.ml.createContext(e);return this.mlContextCache.push({options:e,mlContext:r}),r}}registerMLContext(e,t){this.mlContextBySessionId.set(e,t);let r=this.sessionIdsByMLContext.get(t);r||(r=new Set,this.sessionIdsByMLContext.set(t,r)),r.add(e),this.temporaryGraphInputs.length>0&&(this.sessionGraphInputs.set(e,this.temporaryGraphInputs),this.temporaryGraphInputs=[]),this.temporaryGraphOutputs.length>0&&(this.sessionGraphOutputs.set(e,this.temporaryGraphOutputs),this.temporaryGraphOutputs=[])}onReleaseSession(e){this.sessionGraphInputs.delete(e),this.sessionGraphOutputs.delete(e);let t=this.mlContextBySessionId.get(e);if(!t)return;this.tensorManager.releaseTensorsForSession(e),this.mlContextBySessionId.delete(e);let r=this.sessionIdsByMLContext.get(t);if(r.delete(e),r.size===0){this.sessionIdsByMLContext.delete(t);let i=this.mlContextCache.findIndex(a=>a.mlContext===t);i!==-1&&this.mlContextCache.splice(i,1)}}getMLContext(e){return this.mlContextBySessionId.get(e)}reserveTensorId(){return this.tensorManager.reserveTensorId()}releaseTensorId(e){le("verbose",()=>`[WebNN] releaseTensorId {tensorId: ${e}}`),this.tensorManager.releaseTensorId(e)}async ensureTensor(e,t,r,i,a){let n=Yt.get(r);if(!n)throw new Error(`Unsupported ONNX data type: ${r}`);return this.tensorManager.ensureTensor(e??this.currentSessionId,t,n,i,a)}async createTemporaryTensor(e,t,r){le("verbose",()=>`[WebNN] createTemporaryTensor {onnxDataType: ${t}, shape: ${r}}`);let i=Yt.get(t);if(!i)throw new Error(`Unsupported ONNX data type: ${t}`);let a=this.tensorManager.reserveTensorId();await this.tensorManager.ensureTensor(e,a,i,r,!1);let n=this.temporarySessionTensorIds.get(e);return n?n.push(a):this.temporarySessionTensorIds.set(e,[a]),a}uploadTensor(e,t){if(!me().shouldTransferToMLTensor)throw new Error("Trying to upload to a MLTensor while shouldTransferToMLTensor is false");le("verbose",()=>`[WebNN] uploadTensor {tensorId: ${e}, data: ${t.byteLength}}`),this.tensorManager.upload(e,t)}async downloadTensor(e,t){return this.tensorManager.download(e,t)}createMLTensorDownloader(e,t){return async()=>{let r=await this.tensorManager.download(e);return Pa(r,t)}}registerMLTensor(e,t,r,i){let a=Yt.get(r);if(!a)throw new Error(`Unsupported ONNX data type: ${r}`);let n=this.tensorManager.registerTensor(e,t,a,i);return le("verbose",()=>`[WebNN] registerMLTensor {tensor: ${t}, dataType: ${a}, dimensions: ${i}} -> {tensorId: ${n}}`),n}registerMLConstant(e,t,r,i,a,n,s=!1){if(!n)throw new Error("External mounted files are not available.");let u=e;e.startsWith("./")&&(u=e.substring(2));let d=n.get(u);if(!d)throw new Error(`File with name ${u} not found in preloaded files.`);if(t+r>d.byteLength)throw new Error("Out of bounds: data offset and length exceed the external file data size.");let p=d.slice(t,t+r).buffer,c;switch(a.dataType){case"float32":c=new Float32Array(p);break;case"float16":c=typeof Float16Array<"u"&&Float16Array.from?new Float16Array(p):new Uint16Array(p);break;case"int32":c=new Int32Array(p);break;case"uint32":c=new Uint32Array(p);break;case"int64":if(s){let f=ma(new Uint8Array(p),"int64");c=new Int32Array(f.buffer),a.dataType="int32"}else c=new BigInt64Array(p);break;case"uint64":c=new BigUint64Array(p);break;case"int8":c=new Int8Array(p);break;case"int4":case"uint4":case"uint8":c=new Uint8Array(p);break;default:throw new Error(`Unsupported data type: ${a.dataType} in creating WebNN Constant from external data.`)}return le("verbose",()=>`[WebNN] registerMLConstant {dataType: ${a.dataType}, shape: ${a.shape}}} ${s?"(Note: it was int64 data type and registered to int32 as workaround)":""}`),i.constant(a,c)}registerGraphInput(e){this.temporaryGraphInputs.push(e)}registerGraphOutput(e){this.temporaryGraphOutputs.push(e)}isGraphInput(e,t){let r=this.sessionGraphInputs.get(e);return r?r.includes(t):!1}isGraphOutput(e,t){let r=this.sessionGraphOutputs.get(e);return r?r.includes(t):!1}isGraphInputOutputTypeSupported(e,t,r=!0){let i=this.mlContextBySessionId.get(e),a=Yt.get(It(t));return typeof a>"u"?!1:r?!!(i!=null&&i.opSupportLimits().input.dataTypes.includes(a)):!!(i!=null&&i.opSupportLimits().output.dataTypes.includes(a))}flush(){}}}),qa=P(()=>{}),Bi,Tr,Ir,ko,To,Ni,ga,Io,mp,Xg=P(()=>{nt(),qa(),Bi=new Map([[64,250],[128,200],[256,200],[512,200],[2048,230],[4096,200],[8192,50],[16384,50],[32768,50],[65536,50],[131072,50],[262144,50],[524288,50],[1048576,50],[2097152,30],[4194304,20],[8388608,10],[12582912,10],[16777216,10],[26214400,15],[33554432,22],[44236800,2],[58982400,6],[67108864,6],[134217728,6],[167772160,6]]),Tr=[],Ir=e=>Math.ceil(Number(e)/16)*16,ko=e=>{for(let t=0;t<Tr.length;t++){let r=Tr[t];if(e<=r)return r}return Math.ceil(e/16)*16},To=1,Ni=()=>To++,ga=async(e,t,r,i)=>{let a=Ir(r),n=e.device.createBuffer({size:a,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let s=e.getCommandEncoder();e.endComputePass(),s.copyBufferToBuffer(t,0,n,0,a),e.flush(),await n.mapAsync(GPUMapMode.READ);let u=n.getMappedRange();if(i){let d=i();return d.set(new Uint8Array(u,0,r)),d}else return new Uint8Array(u.slice(0,r))}finally{n.destroy()}},Io=class{constructor(e){this.backend=e,this.storageCache=new Map,this.freeBuffers=new Map,this.freeUniformBuffers=new Map,this.buffersPending=[],this.capturedPendingBuffers=new Map;for(let[t]of Bi)Tr.push(t),this.freeBuffers.set(t,[]),this.freeUniformBuffers.set(t,[]);this.sessionCount=0}upload(e,t){let r=t.buffer,i=t.byteOffset,a=t.byteLength,n=Ir(a),s=this.storageCache.get(e);if(!s)throw new Error("gpu data for uploading does not exist");if(Number(s.originalSize)!==a)throw new Error(`inconsistent data size. gpu data size=${s.originalSize}, data size=${a}`);let u=this.backend.device.createBuffer({mappedAtCreation:!0,size:n,usage:GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC}),d=u.getMappedRange();new Uint8Array(d).set(new Uint8Array(r,i,a)),u.unmap();let p=this.backend.device.createCommandEncoder();p.copyBufferToBuffer(u,0,s.gpuData.buffer,0,n),this.backend.device.queue.submit([p.finish()]),u.destroy(),le("verbose",()=>`[WebGPU] GpuDataManager.upload(id=${e})`)}memcpy(e,t){let r=this.storageCache.get(e);if(!r)throw new Error("source gpu data for memcpy does not exist");let i=this.storageCache.get(t);if(!i)throw new Error("destination gpu data for memcpy does not exist");if(r.originalSize!==i.originalSize)throw new Error("inconsistent source and destination gpu data size");let a=Ir(r.originalSize),n=this.backend.getCommandEncoder();this.backend.endComputePass(),n.copyBufferToBuffer(r.gpuData.buffer,0,i.gpuData.buffer,0,a)}registerExternalBuffer(e,t,r){let i;if(r){if(i=r[0],e===r[1])return le("verbose",()=>`[WebGPU] GpuDataManager.registerExternalBuffer(size=${t}) => id=${i}, buffer is the same, skip.`),i;if(this.backend.capturedCommandList.has(this.backend.currentSessionId))throw new Error(`Registering a different external buffer under graph capture mode is not supported yet.
             Please use the previous external buffer!`)}else i=Ni();return this.storageCache.set(i,{gpuData:{id:i,type:0,buffer:e},originalSize:t}),le("verbose",()=>`[WebGPU] GpuDataManager.registerExternalBuffer(size=${t}) => id=${i}, registered.`),i}unregisterExternalBuffer(e){e!==void 0&&(this.storageCache.delete(e),le("verbose",()=>`[WebGPU] GpuDataManager.unregisterExternalBuffer() => id=${e}`))}create(e,t=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST){let r=ko(e),i,a=(t&GPUBufferUsage.STORAGE)===GPUBufferUsage.STORAGE,n=(t&GPUBufferUsage.UNIFORM)===GPUBufferUsage.UNIFORM;if(a||n){let u=(a?this.freeBuffers:this.freeUniformBuffers).get(r);u?u.length>0?i=u.pop():i=this.backend.device.createBuffer({size:r,usage:t}):i=this.backend.device.createBuffer({size:r,usage:t})}else i=this.backend.device.createBuffer({size:r,usage:t});let s={id:Ni(),type:0,buffer:i};return this.storageCache.set(s.id,{gpuData:s,originalSize:Number(e)}),le("verbose",()=>`[WebGPU] GpuDataManager.create(size=${e}) => id=${s.id}`),s}get(e){var t;return(t=this.storageCache.get(e))==null?void 0:t.gpuData}release(e){let t=typeof e=="bigint"?Number(e):e,r=this.storageCache.get(t);if(!r){if(this.storageCache.size===0)return 0;throw new Error("releasing data does not exist")}return le("verbose",()=>`[WebGPU] GpuDataManager.release(id=${t}), gpuDataId=${r.gpuData.id}`),this.storageCache.delete(t),this.buffersPending.push(r.gpuData.buffer),r.originalSize}async download(e,t){let r=this.storageCache.get(Number(e));if(!r)throw new Error("data does not exist");await ga(this.backend,r.gpuData.buffer,r.originalSize,t)}refreshPendingBuffers(){if(this.buffersPending.length!==0)if(this.backend.sessionStatus==="default"){for(let e of this.buffersPending){let t=Bi.get(e.size);if((e.usage&GPUBufferUsage.STORAGE)===GPUBufferUsage.STORAGE){let r=this.freeBuffers.get(e.size)||[];t===void 0||r.length>=t?e.destroy():r.push(e)}else if((e.usage&GPUBufferUsage.UNIFORM)===GPUBufferUsage.UNIFORM){let r=this.freeUniformBuffers.get(e.size)||[];t===void 0||r.length>=t?e.destroy():r.push(e)}else e.destroy()}this.buffersPending=[]}else{let e=this.capturedPendingBuffers.get(this.backend.currentSessionId);e||(e=[],this.capturedPendingBuffers.set(this.backend.currentSessionId,e));for(let t of this.buffersPending)e.push(t);this.buffersPending=[]}}dispose(){this.freeBuffers.forEach(e=>{e.forEach(t=>{t.destroy()})}),this.freeUniformBuffers.forEach(e=>{e.forEach(t=>{t.destroy()})}),this.storageCache.forEach(e=>{e.gpuData.buffer.destroy()}),this.capturedPendingBuffers.forEach(e=>{e.forEach(t=>{t.destroy()})}),this.storageCache=new Map,this.freeBuffers=new Map,this.freeUniformBuffers=new Map,this.capturedPendingBuffers=new Map}onCreateSession(){this.sessionCount+=1}onReleaseSession(e){let t=this.capturedPendingBuffers.get(e);t&&(t.forEach(r=>{r.destroy()}),this.capturedPendingBuffers.delete(e)),this.sessionCount-=1,this.sessionCount===0&&(le("warning",()=>"[WebGPU] Clearing webgpu buffer cache"),this.storageCache.forEach(r=>{r.gpuData.buffer.destroy()}),this.storageCache=new Map)}},mp=(...e)=>new Io(...e)}),Eo,ce,ve=P(()=>{Eo=class{constructor(e){Object.assign(this,e)}get cacheKey(){return this.key||(this.key=Object.getOwnPropertyNames(this).sort().map(e=>`${this[e]}`).join(";")),this.key}},ce=e=>new Eo(e)}),Lt,Er,ke,Ce,Q,$e,ya,qt,gt,Z,Xt,M,j,gp,Wa,zo,yp,ae=P(()=>{J(),ie(),Lt=64,Er=(e,t)=>{if(t===3)throw new Error("vec3 has same alignment as vec4, use vec4 instead");switch(Number(e)){case 10:return t>1?`vec${t}<f16>`:"f16";case 1:return t>1?`vec${t}<f32>`:"f32";case 6:return t>1?`vec${t}<i32>`:"i32";case 12:return t>1?`vec${t}<u32>`:"u32";case 7:if(t>1)throw new Error("currently not supported vecX of uint64 yet");return["vec2<u32>","i32"];case 13:if(t>1)throw new Error("currently not supported vecX of uint64 yet");return["vec2<u32>","u32"];case 9:if(t!==4)throw new Error("bool must be vec4");return["u32","vec4<bool>"];case 22:return"i32";case 21:return"u32";default:throw new Error(`Unknown data type: ${e}`)}},ke=(e,t=1)=>{let r=Er(e,t);return typeof r=="string"?r:r[0]},Ce=(e,t=1)=>{let r=Er(e,t);return typeof r=="string"?r:r[1]},Q=(...e)=>{let t=[];return e.forEach(r=>{r.length!==0&&t.push({type:12,data:r},{type:12,data:B.computeStrides(r)})}),t},$e=e=>e%4===0?4:e%2===0?2:1,ya=(e="f32",t,r="0")=>!t||t===1?`${e}(${r})`:`vec${t}<${e}>(${r})`,qt=(e,t,r)=>e==="f32"?r:t===1?`f32(${r})`:`vec${t}<f32>(${r})`,gt=(e,t)=>t===4?`(${e}.x + ${e}.y + ${e}.z + ${e}.w)`:t===2?`(${e}.x + ${e}.y)`:t===3?`(${e}.x + ${e}.y + ${e}.z)`:e,Z=(e,t,r,i)=>e.startsWith("uniforms.")&&r>4?typeof t=="string"?i==="f16"?`${e}[(${t}) / 8][(${t}) % 8 / 4][(${t}) % 8 % 4]`:`${e}[(${t}) / 4][(${t}) % 4]`:i==="f16"?`${e}[${Math.floor(t/8)}][${Math.floor(t%8/4)}][${t%8%4}]`:`${e}[${Math.floor(t/4)}][${t%4}]`:r>1?`${e}[${t}]`:e,Xt=(e,t,r,i,a)=>{let n=typeof r=="number",s=n?r:r.length,u=[...new Array(s).keys()],d=s<2?"u32":s<=4?`vec${s}<u32>`:`array<u32, ${s}>`,p=Er(t,a),c=typeof p=="string"?p:p[1],f=typeof p=="string"?p:p[0],g={indices:d,value:c,storage:f,tensor:t},_=D=>typeof D=="string"?D:`${D}u`,w={offsetToIndices:!1,indicesToOffset:!1,broadcastedIndicesToOffset:!1,set:!1,setByIndices:!1,get:!1,getByIndices:!1},b=n?"uniforms.":"",S=`${b}${e}_shape`,v=`${b}${e}_strides`,$="";for(let D=0;D<s-1;D++)$+=`
    let dim${D} = current / ${Z(v,D,s)};
    let rest${D} = current % ${Z(v,D,s)};
    indices[${D}] = dim${D};
    current = rest${D};
    `;$+=`indices[${s-1}] = current;`;let I=s<2?"":`
  fn o2i_${e}(offset: u32) -> ${g.indices} {
    var indices: ${g.indices};
    var current = offset;
    ${$}
    return indices;
  }`,T=D=>(w.offsetToIndices=!0,s<2?D:`o2i_${e}(${D})`),E=[];if(s>=2)for(let D=s-1;D>=0;D--)E.push(`${Z(v,D,s)} * (indices[${D}])`);let A=s<2?"":`
  fn i2o_${e}(indices: ${g.indices}) -> u32 {
    return ${E.join("+")};
  }`,C=D=>(w.indicesToOffset=!0,s<2?D:`i2o_${e}(${D})`),O=(...D)=>s===0?"0u":`${g.indices}(${D.map(_).join(",")})`,U=(D,L)=>s<2?`${D}`:`${Z(D,L,s)}`,x=(D,L,K)=>s<2?`${D}=${K};`:`${Z(D,L,s)}=${K};`,Y={},G=(D,L)=>{w.broadcastedIndicesToOffset=!0;let K=`${L.name}broadcastedIndicesTo${e}Offset`;if(K in Y)return`${K}(${D})`;let re=[];for(let ze=s-1;ze>=0;ze--){let Ke=L.indicesGet("outputIndices",ze+L.rank-s);re.push(`${U(v,ze)} * (${Ke} % ${U(S,ze)})`)}return Y[K]=`fn ${K}(outputIndices: ${L.type.indices}) -> u32 {
             return ${re.length>0?re.join("+"):"0u"};
           }`,`${K}(${D})`},V=(D,L)=>(()=>{if(g.storage===g.value)return`${e}[${D}]=${L};`;if(g.storage==="vec2<u32>"&&g.value==="i32")return`${e}[${D}]=vec2<u32>(u32(${L}), select(0u, 0xFFFFFFFFu, ${L} < 0));`;if(g.storage==="vec2<u32>"&&g.value==="u32")return`${e}[${D}]=vec2<u32>(u32(${L}), 0u);`;if(g.storage==="u32"&&g.value==="vec4<bool>")return`${e}[${D}]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(${L}));`;throw new Error(`not supported combination of storage type ${g.storage} and value type ${g.value} yet`)})(),te=D=>(()=>{if(g.storage===g.value)return`${e}[${D}]`;if(g.storage==="vec2<u32>"&&g.value==="i32")return`i32(${e}[${D}].x)`;if(g.storage==="vec2<u32>"&&g.value==="u32")return`u32(${e}[${D}].x)`;if(g.storage==="u32"&&g.value==="vec4<bool>")return`vec4<bool>(bool(${e}[${D}] & 0xFFu), bool(${e}[${D}] & 0xFF00u), bool(${e}[${D}] & 0xFF0000u), bool(${e}[${D}] & 0xFF000000u))`;throw new Error(`not supported combination of storage type ${g.storage} and value type ${g.value} yet`)})(),ee=s<2?"":`
  fn get_${e}ByIndices(indices: ${g.indices}) -> ${c} {
    return ${te(`i2o_${e}(indices)`)};
  }`,F=s<2?"":(()=>{let D=u.map(K=>`d${K}: u32`).join(", "),L=u.map(K=>`d${K}`).join(", ");return`
  fn get_${e}(${D}) -> ${c} {
    return get_${e}ByIndices(${O(L)});
  }`})(),R=(...D)=>{if(D.length!==s)throw new Error(`indices length must be ${s}`);let L=D.map(_).join(",");return s===0?te("0u"):s===1?te(L[0]):(w.get=!0,w.getByIndices=!0,w.indicesToOffset=!0,`get_${e}(${L})`)},q=D=>s<2?te(D):(w.getByIndices=!0,w.indicesToOffset=!0,`get_${e}ByIndices(${D})`),X=s<2?"":`
  fn set_${e}ByIndices(indices: ${g.indices}, value: ${c}) {
    ${V(`i2o_${e}(indices)`,"value")}
  }`,_e=s<2?"":(()=>{let D=u.map(K=>`d${K}: u32`).join(", "),L=u.map(K=>`d${K}`).join(", ");return`
  fn set_${e}(${D}, value: ${c}) {
    set_${e}ByIndices(${O(L)}, value);
  }`})();return{impl:()=>{let D=[],L=!1;return w.offsetToIndices&&(D.push(I),L=!0),w.indicesToOffset&&(D.push(A),L=!0),w.broadcastedIndicesToOffset&&(Object.values(Y).forEach(K=>D.push(K)),L=!0),w.set&&(D.push(_e),L=!0),w.setByIndices&&(D.push(X),L=!0),w.get&&(D.push(F),L=!0),w.getByIndices&&(D.push(ee),L=!0),!n&&L&&D.unshift(`const ${S} = ${g.indices}(${r.join(",")});`,`const ${v} = ${g.indices}(${B.computeStrides(r).join(",")});`),D.join(`
`)},type:g,offsetToIndices:T,indicesToOffset:C,broadcastedIndicesToOffset:G,indices:O,indicesGet:U,indicesSet:x,set:(...D)=>{if(D.length!==s+1)throw new Error(`indices length must be ${s}`);let L=D[s];if(typeof L!="string")throw new Error("value must be string");let K=D.slice(0,s).map(_).join(",");return s===0?V("0u",L):s===1?V(K[0],L):(w.set=!0,w.setByIndices=!0,w.indicesToOffset=!0,`set_${e}(${K}, ${L})`)},setByOffset:V,setByIndices:(D,L)=>s<2?V(D,L):(w.setByIndices=!0,w.indicesToOffset=!0,`set_${e}ByIndices(${D}, ${L});`),get:R,getByOffset:te,getByIndices:q,usage:i,name:e,strides:v,shape:S,rank:s}},M=(e,t,r,i=1)=>Xt(e,t,r,"input",i),j=(e,t,r,i=1)=>Xt(e,t,r,"output",i),gp=(e,t,r)=>Xt(e,t,r,"atomicOutput",1),Wa=(e,t,r,i=1)=>Xt(e,t,r,"internal",i),zo=class{constructor(e,t){this.normalizedDispatchGroup=e,this.limits=t,this.internalVariables=[],this.variables=[],this.uniforms=[],this.variableIndex=0}guardAgainstOutOfBoundsWorkgroupSizes(e){return`if (global_idx >= ${typeof e=="number"?`${e}u`:e}) { return; }`}mainStart(e=Lt){let t=typeof e=="number"?e:e[0],r=typeof e=="number"?1:e[1],i=typeof e=="number"?1:e[2];if(t>this.limits.maxComputeWorkgroupSizeX||r>this.limits.maxComputeWorkgroupSizeY||i>this.limits.maxComputeWorkgroupSizeZ)throw new Error(`workgroup size [${t}, ${r}, ${i}] exceeds the maximum workgroup size [${this.limits.maxComputeWorkgroupSizeX}, ${this.limits.maxComputeWorkgroupSizeY}, ${this.limits.maxComputeWorkgroupSizeZ}].`);if(t*r*i>this.limits.maxComputeInvocationsPerWorkgroup)throw new Error(`workgroup size [${t}, ${r}, ${i}] exceeds the maximum workgroup invocations ${this.limits.maxComputeInvocationsPerWorkgroup}.`);let a=this.normalizedDispatchGroup[1]===1&&this.normalizedDispatchGroup[2]===1,n=a?`@builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_index) local_idx : u32,
    @builtin(local_invocation_id) local_id : vec3<u32>`:`@builtin(global_invocation_id) global_id : vec3<u32>,
                                             @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(local_invocation_index) local_idx : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>`,s=a?`let global_idx = global_id.x;
         let workgroup_index = workgroup_id.x;`:`let workgroup_index = workgroup_id.z * num_workgroups[0] * num_workgroups[1] +
             workgroup_id.y * num_workgroups[0] + workgroup_id.x;
         let global_idx = workgroup_index * ${t*r*i}u + local_idx;`;return`@compute @workgroup_size(${t}, ${r}, ${i})
  fn main(${n}) {
    ${s}
  `}appendVariableUniforms(e){e.rank!==0&&(e.shape.startsWith("uniforms.")&&this.uniforms.push({name:e.shape.replace("uniforms.",""),type:"u32",length:e.rank}),e.strides.startsWith("uniforms.")&&this.uniforms.push({name:e.strides.replace("uniforms.",""),type:"u32",length:e.rank}))}declareVariable(e,t){if(e.usage==="internal")throw new Error("cannot use internal variable with declareVariable(). use registerInternalVariables() instead.");this.variables.push(e),this.appendVariableUniforms(e);let r=e.usage==="input"?"read":"read_write",i=e.usage==="atomicOutput"?"atomic<i32>":e.type.storage;return`@group(0) @binding(${t}) var<storage, ${r}> ${e.name}: array<${i}>;`}declareVariables(...e){return e.map(t=>this.declareVariable(t,this.variableIndex++)).join(`
`)}registerInternalVariable(e){if(e.usage!=="internal")throw new Error("cannot use input or output variable with registerInternalVariable(). use declareVariables() instead.");this.internalVariables.push(e),this.appendVariableUniforms(e)}registerInternalVariables(...e){return e.forEach(t=>this.registerInternalVariable(t)),this}registerUniform(e,t,r=1){return this.uniforms.push({name:e,type:t,length:r}),this}registerUniforms(e){return this.uniforms=this.uniforms.concat(e),this}uniformDeclaration(){if(this.uniforms.length===0)return"";let e=[];for(let{name:t,type:r,length:i}of this.uniforms)if(i&&i>4)r==="f16"?e.push(`@align(16) ${t}:array<mat2x4<${r}>, ${Math.ceil(i/8)}>`):e.push(`${t}:array<vec4<${r}>, ${Math.ceil(i/4)}>`);else{let a=i==null||i===1?r:`vec${i}<${r}>`;e.push(`${t}:${a}`)}return`
      struct Uniforms { ${e.join(", ")} };
      @group(0) @binding(${this.variableIndex}) var<uniform> uniforms: Uniforms;`}get additionalImplementations(){return this.uniformDeclaration()+this.variables.map(e=>e.impl()).join(`
`)+this.internalVariables.map(e=>e.impl()).join(`
`)}get variablesInfo(){if(this.uniforms.length===0)return;let e=t=>[12,10,1,6][["u32","f16","f32","i32"].indexOf(t)];return this.uniforms.map(t=>[e(t.type),t.length??1])}},yp=(e,t)=>new zo(e,t)}),Co,Di,Ao,Oo,Ro,Bo,De,_p,wp,yt=P(()=>{J(),ie(),ve(),ae(),Co=(e,t)=>{if(!e||e.length!==1)throw new Error("Transpose requires 1 input.");if(t.length!==0&&t.length!==e[0].dims.length)throw new Error(`perm size ${t.length} does not match input rank ${e[0].dims.length}`)},Di=(e,t)=>t.length!==0?t:[...new Array(e).keys()].reverse(),Ao=(e,t)=>B.sortBasedOnPerm(e,Di(e.length,t)),Oo=(e,t,r,i)=>{let a=`fn perm(i: ${i.type.indices}) -> ${r.type.indices} {
    var a: ${r.type.indices};`;for(let n=0;n<t;++n)a+=`a[${e[n]}]=i[${n}];`;return a+="return a;}"},Ro=(e,t)=>{let r=[],i=[];for(let a=0;a<e.length;++a)e[a]!==1&&r.push(e[a]),e[t[a]]!==1&&i.push(t[a]);return{newShape:r,newPerm:i}},Bo=(e,t)=>{let r=0;for(let i=0;i<e.length;++i)if(t[e[i]]!==1){if(e[i]<r)return!1;r=e[i]}return!0},De=(e,t)=>{let r=e.dataType,i=e.dims.length,a=Di(i,t),n=Ao(e.dims,a),s=e.dims,u=n,d=i<2||Bo(a,e.dims),p;if(d)return p=w=>{let b=M("input",r,s,4),S=j("output",r,u,4);return`
  ${w.registerUniform("output_size","u32").declareVariables(b,S)}
  ${w.mainStart()}
    ${w.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    output[global_idx] = input[global_idx];
  }`},{name:"TransposeCopy",shaderCache:{inputDependencies:["type"]},getRunData:()=>{let w=B.size(n);return{outputs:[{dims:n,dataType:e.dataType}],dispatchGroup:{x:Math.ceil(w/64/4)},programUniforms:[{type:12,data:Math.ceil(w/4)}]}},getShaderSource:p};let{newShape:c,newPerm:f}=Ro(e.dims,a),g=B.areEqual(f,[2,3,1]),_=B.areEqual(f,[3,1,2]);if(c.length===2||g||_){s=g?[c[0],c[1]*c[2]]:_?[c[0]*c[1],c[2]]:c,u=[s[1],s[0]];let w=16;return p=b=>{let S=M("a",r,s.length),v=j("output",r,u.length);return`
  ${b.registerUniform("output_size","u32").declareVariables(S,v)}
  var<workgroup> tile : array<array<${v.type.value}, ${w+1}>, ${w}>;
  ${b.mainStart([w,w,1])}
    let stride = (uniforms.output_shape[1] - 1) / ${w} + 1;
    let workgroup_id_x = workgroup_index % stride;
    let workgroup_id_y = workgroup_index / stride;
    let input_col = workgroup_id_y * ${w}u + local_id.x;
    let input_row = workgroup_id_x * ${w}u + local_id.y;
    if (input_row < uniforms.a_shape[0] && input_col < uniforms.a_shape[1]) {
      tile[local_id.y][local_id.x] = ${S.getByIndices(`${S.type.indices}(input_row, input_col)`)};
    }
    workgroupBarrier();

    let output_col = workgroup_id_x * ${w}u + local_id.x;
    let output_row = workgroup_id_y * ${w}u + local_id.y;
    if (output_row < uniforms.output_shape[0] && output_col < uniforms.output_shape[1]) {
      ${v.setByIndices(`${v.type.indices}(output_row, output_col)`,"tile[local_id.x][local_id.y]")}
    }
  }`},{name:"TransposeShared",shaderCache:{inputDependencies:["type"]},getRunData:()=>{let b=B.size(n);return{outputs:[{dims:n,dataType:e.dataType}],dispatchGroup:{x:Math.ceil(u[1]/w),y:Math.ceil(u[0]/w)},programUniforms:[{type:12,data:b},...Q(s,u)]}},getShaderSource:p}}return p=w=>{let b=M("a",r,s.length),S=j("output",r,u.length);return`
  ${w.registerUniform("output_size","u32").declareVariables(b,S)}

  ${Oo(a,i,b,S)}

  ${w.mainStart()}
    ${w.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}

    let indices = ${S.offsetToIndices("global_idx")};
    let aIndices = perm(indices);

    ${S.setByOffset("global_idx",b.getByIndices("aIndices"))}
  }`},{name:"Transpose",shaderCache:{hint:`${t}`,inputDependencies:["rank"]},getRunData:()=>{let w=B.size(n);return{outputs:[{dims:n,dataType:e.dataType}],dispatchGroup:{x:Math.ceil(w/64)},programUniforms:[{type:12,data:w},...Q(s,u)]}},getShaderSource:p}},_p=(e,t)=>{Co(e.inputs,t.perm),e.compute(De(e.inputs[0],t.perm))},wp=e=>ce({perm:e.perm})}),No,Do,Mo,Uo,Po,qo,Wo,Lo,Vo,Go,We,bp,$p,vp,xp,Sp,kp,Tp,Ip,Ep,zp,Jg=P(()=>{J(),ie(),ae(),La(),yt(),No={max:"select(bestValue, candidate, candidate > bestValue)",min:"select(bestValue, candidate, candidate < bestValue)",mean:"bestValue + candidate",sum:"bestValue + candidate",prod:"bestValue * candidate",sumSquare:"bestValue + candidate * candidate",logSumExp:"bestValue + exp(candidate)",l1:"bestValue + abs(candidate)",l2:"bestValue + candidate * candidate",logSum:"bestValue + candidate"},Do={max:"select(bestValue, candidate, candidate > bestValue)",min:"select(bestValue, candidate, candidate < bestValue)",mean:"bestValue + candidate",sum:"bestValue + candidate",prod:"bestValue * candidate",sumSquare:"bestValue + candidate",logSumExp:"bestValue + candidate",l1:"bestValue + candidate",l2:"bestValue + candidate",logSum:"bestValue + candidate"},Mo={max:"_A[offset]",min:"_A[offset]",mean:"0",sum:"0",prod:"1",sumSquare:"0",logSumExp:"0",l1:"0",l2:"0",logSum:"0"},Uo={max:"bestValue",min:"bestValue",sum:"bestValue",prod:"bestValue",sumSquare:"bestValue",logSumExp:"log(bestValue)",l1:"bestValue",l2:"sqrt(bestValue)",logSum:"log(bestValue)"},Po=(e,t)=>{let r=[];for(let i=t-e;i<t;++i)r.push(i);return r},qo=(e,t)=>{let r=[],i=e.length;for(let n=0;n<i;n++)t.indexOf(n)===-1&&r.push(e[n]);let a=t.map(n=>e[n]);return[r,a]},Wo=(e,t)=>{let r=e.length+t.length,i=[],a=0;for(let n=0;n<r;n++)t.indexOf(n)===-1?i.push(e[a++]):i.push(1);return i},Lo=(e,t)=>{for(let r=0;r<e.length;++r)if(e[e.length-r-1]!==t-1-r)return!1;return!0},Vo=(e,t)=>{let r=[];if(!Lo(e,t)){for(let i=0;i<t;++i)e.indexOf(i)===-1&&r.push(i);e.forEach(i=>r.push(i))}return r},Go=(e,t,r,i,a,n,s)=>{let u=r[0].dims,d=B.size(n),p=B.size(s),c=M("_A",r[0].dataType,u),f=j("output",a,n),g=64;d===1&&(g=256);let _=`
          var<workgroup> aBestValues : array<f32, ${g}>;
       `,w=b=>`
        ${b.registerUniform("reduceSize","u32").declareVariables(c,f)}
        ${_}
        fn DIV_CEIL(a : u32, b : u32) -> u32 {
          return ((a - 1u) / b + 1u);
         }
         ${b.mainStart(g)}

          let outputIndex = global_idx / ${g};
          let offset = outputIndex * uniforms.reduceSize;

          var bestValue = f32(${Mo[i]});
          let Length = uniforms.reduceSize;
          for (var k = local_idx; k < Length; k = k + ${g}) {
           let candidate = f32(${c.getByOffset("offset + k")});
           bestValue = ${No[i]};
          }
          aBestValues[local_idx] = bestValue;
          workgroupBarrier();

         var reduceSize = min(Length, ${g}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (local_idx < currentSize) {
            let candidate = aBestValues[local_idx + interval];
            bestValue = ${Do[i]};
            aBestValues[local_idx] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (local_idx == 0u) {
          ${f.setByOffset("outputIndex",`${i==="mean"?`${f.type.storage}(bestValue / f32(uniforms.reduceSize))`:`${f.type.storage}(${Uo[i]})`}`)};
         }
        }`;return{name:e,shaderCache:{hint:`${t};${g}`,inputDependencies:["type"]},getShaderSource:w,getRunData:()=>({outputs:[{dims:n,dataType:a}],dispatchGroup:{x:d},programUniforms:[{type:12,data:p}]})}},We=(e,t,r,i)=>{let a=e.inputs.length===1?r:_a(e.inputs,r),n=a.axes;n.length===0&&!a.noopWithEmptyAxes&&(n=e.inputs[0].dims.map((_,w)=>w));let s=B.normalizeAxes(n,e.inputs[0].dims.length),u=s,d=e.inputs[0],p=Vo(u,e.inputs[0].dims.length);p.length>0&&(d=e.compute(De(e.inputs[0],p),{inputs:[0],outputs:[-1]})[0],u=Po(u.length,d.dims.length));let[c,f]=qo(d.dims,u),g=c;a.keepDims&&(g=Wo(c,s)),e.compute(Go(t,a.cacheKey,[d],i,e.inputs[0].dataType,g,f),{inputs:[d]})},bp=(e,t)=>{We(e,"ReduceMeanShared",t,"mean")},$p=(e,t)=>{We(e,"ReduceL1Shared",t,"l1")},vp=(e,t)=>{We(e,"ReduceL2Shared",t,"l2")},xp=(e,t)=>{We(e,"ReduceLogSumExpShared",t,"logSumExp")},Sp=(e,t)=>{We(e,"ReduceMaxShared",t,"max")},kp=(e,t)=>{We(e,"ReduceMinShared",t,"min")},Tp=(e,t)=>{We(e,"ReduceProdShared",t,"prod")},Ip=(e,t)=>{We(e,"ReduceSumShared",t,"sum")},Ep=(e,t)=>{We(e,"ReduceSumSquareShared",t,"sumSquare")},zp=(e,t)=>{We(e,"ReduceLogSumShared",t,"logSum")}}),Le,Ho,Lr,_a,Ve,Fo,jo,Ko,Zo,Qo,Yo,Xo,Jo,eu,tu,Ge,Cp,Ap,Op,Rp,Bp,Np,Dp,Mp,Up,Pp,La=P(()=>{J(),ie(),ve(),ae(),Jg(),Le=e=>{if(!e||e.length===0||e.length>2)throw new Error("Reduce op requires 1 or 2 inputs.");if(e.length===2&&e[1].dims.length!==1)throw new Error("Invalid axes input dims.")},Ho=e=>["","",`var value = ${e.getByIndices("input_indices")};`,""],Lr=(e,t,r,i,a,n,s=!1,u=!1)=>{let d=[],p=r[0].dims,c=p.length,f=B.normalizeAxes(a,c),g=!u&&f.length===0;p.forEach((b,S)=>{g||f.indexOf(S)>=0?s&&d.push(1):d.push(b)});let _=d.length,w=B.size(d);return{name:e,shaderCache:t,getShaderSource:b=>{let S=[],v=M("_A",r[0].dataType,c),$=j("output",n,_),I=i(v,$,f),T=I[2];for(let E=0,A=0;E<c;E++)g||f.indexOf(E)>=0?(s&&A++,T=`for(var j${E}: u32 = 0; j${E} < ${p[E]}; j${E}++) {
                  ${I[2].includes("last_index")?`let last_index = j${E};`:""}
                  ${v.indicesSet("input_indices",E,`j${E}`)}
                  ${T}
                }`):(S.push(`${v.indicesSet("input_indices",E,$.indicesGet("output_indices",A))};`),A++);return`

        ${b.registerUniform("output_size","u32").declareVariables(v,$)}

        ${b.mainStart()}
          ${b.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
          var input_indices: ${v.type.indices};
          let output_indices = ${$.offsetToIndices("global_idx")};

          ${S.join(`
`)}
          ${I[0]}       // init ops for reduce max/min
          ${I[1]}
          ${T}
          ${I[3]}
          ${I.length===4?$.setByOffset("global_idx","value"):I.slice(4).join(`
`)}
        }`},getRunData:()=>({outputs:[{dims:d,dataType:n}],dispatchGroup:{x:Math.ceil(w/64)},programUniforms:[{type:12,data:w},...Q(p,d)]})}},_a=(e,t)=>{let r=[];return e[1].dims[0]>0&&e[1].getBigInt64Array().forEach(i=>r.push(Number(i))),ce({axes:r,keepDims:t.keepDims,noopWithEmptyAxes:t.noopWithEmptyAxes})},Ve=(e,t,r,i)=>{let a=e.inputs,n=a.length===1?r:_a(a,r);e.compute(Lr(t,{hint:n.cacheKey,inputDependencies:["rank"]},[a[0]],n.noopWithEmptyAxes&&n.axes.length===0?Ho:i,n.axes,a[0].dataType,n.keepDims,n.noopWithEmptyAxes),{inputs:[0]})},Fo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceLogSum",t,(r,i)=>[`var value = ${i.type.storage}(0);`,"",`value += ${r.getByIndices("input_indices")};`,"value = log(value);"])},jo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceL1",t,(r,i)=>[`var value = ${i.type.storage}(0);`,"",`value += abs(${r.getByIndices("input_indices")});`,""])},Ko=(e,t)=>{Le(e.inputs),Ve(e,"ReduceL2",t,(r,i)=>[`var t = ${i.type.value}(0); var value = ${i.type.value}(0);`,"",`t = ${r.getByIndices("input_indices")}; value += (t * t);`,"value = sqrt(value);"])},Zo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceLogSumExp",t,(r,i)=>[`var value = ${i.type.storage}(0);`,"",`value += exp(${r.getByIndices("input_indices")});`,"value = log(value);"])},Qo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceMax",t,(r,i,a)=>{let n=[];for(let s=0;s<r.rank;s++)(a.indexOf(s)>=0||a.length===0)&&n.push(r.indicesSet("input_indices",s,0));return[`${n.join(`
`)}`,`var value = ${r.getByIndices("input_indices")};`,`value = max(value, ${r.getByIndices("input_indices")});`,""]})},Yo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceMean",t,(r,i,a)=>{let n=1;for(let s=0;s<r.rank;s++)(a.indexOf(s)>=0||a.length===0)&&(n*=e.inputs[0].dims[s]);return["var sum = f32(0);","",`sum += f32(${r.getByIndices("input_indices")});`,`let value = ${i.type.value}(sum / ${n});`]})},Xo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceMin",t,(r,i,a)=>{let n=[];for(let s=0;s<r.rank;s++)(a.indexOf(s)>=0||a.length===0)&&n.push(`input_indices[${s}] = 0;`);return[`${n.join(`
`)}`,`var value = ${r.getByIndices("input_indices")};`,`value = min(value, ${r.getByIndices("input_indices")});`,""]})},Jo=(e,t)=>{Le(e.inputs),Ve(e,"ReduceProd",t,(r,i)=>[`var value = ${i.type.storage}(1);`,"",`value *= ${r.getByIndices("input_indices")};`,""])},eu=(e,t)=>{Le(e.inputs),Ve(e,"ReduceSum",t,(r,i)=>[`var value = ${i.type.storage}(0);`,"",`value += ${r.getByIndices("input_indices")};`,""])},tu=(e,t)=>{Le(e.inputs),Ve(e,"ReduceSumSquare",t,(r,i)=>[`var t = ${i.type.value}(0); var value = ${i.type.value}(0);`,"",`t = ${r.getByIndices("input_indices")}; value += t * t;`,""])},Ge=(e,t,r)=>{if(t.length===0)return r;let i=1,a=1;for(let n=0;n<t.length;n++)t.indexOf(n)===-1?i*=e[n]:a*=e[n];return a<32&&i>1024},Cp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Yo(e,t):bp(e,t)},Ap=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?jo(e,t):$p(e,t)},Op=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Ko(e,t):vp(e,t)},Rp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Zo(e,t):xp(e,t)},Bp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Qo(e,t):Sp(e,t)},Np=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Xo(e,t):kp(e,t)},Dp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Jo(e,t):Tp(e,t)},Mp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?eu(e,t):Ip(e,t)},Up=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?tu(e,t):Ep(e,t)},Pp=(e,t)=>{Ge(e.inputs[0].dims,t.axes,t.noopWithEmptyAxes)?Fo(e,t):zp(e,t)}}),Mi,qp,Wp,wa,e0=P(()=>{J(),ve(),La(),Mi=e=>{if(!e||e.length===0||e.length>2)throw new Error("ArgMinMaxOp op requires 1 or 2 inputs.");if(e[0].dataType!==1)throw new Error("Invalid input type.")},qp=(e,t)=>{Mi(e.inputs);let r=(i,a,n)=>{let s=[];for(let u=0;u<i.rank;u++)(n.indexOf(u)>=0||n.length===0)&&s.push(`input_indices[${u}] = 0;`);return[`${s.join(`
`)}`,`var value = ${i.getByIndices("input_indices")};
var best_index : i32 = 0;`,`if (${i.getByIndices("input_indices")} ${t.selectLastIndex>0?"<=":"<"} value) {
         value = ${i.getByIndices("input_indices")};
         best_index = i32(last_index);
       }`,"",a.setByOffset("global_idx","best_index")]};e.compute(Lr("ArgMin",{hint:t.cacheKey,inputDependencies:["rank"]},[e.inputs[0]],r,[t.axis],7,t.keepDims),{inputs:[0]})},Wp=(e,t)=>{Mi(e.inputs);let r=(i,a,n)=>{let s=[];for(let u=0;u<i.rank;u++)(n.indexOf(u)>=0||n.length===0)&&s.push(`input_indices[${u}] = 0;`);return[`${s.join(`
`)}`,`var value = ${i.getByIndices("input_indices")};
var best_index : i32 = 0;`,`if (${i.getByIndices("input_indices")} ${t.selectLastIndex>0?">=":">"} value) {
         value = ${i.getByIndices("input_indices")};
         best_index = i32(last_index);
       }`,"",a.setByOffset("global_idx","best_index")]};e.compute(Lr("argMax",{hint:t.cacheKey,inputDependencies:["rank"]},[e.inputs[0]],r,[t.axis],7,t.keepDims),{inputs:[0]})},wa=e=>ce(e)}),ru,zr,iu,au,nu,pr,su,Lp,Va=P(()=>{J(),ie(),qa(),ae(),ru=(e,t)=>{let r=e[0],i=e[1],a=e[2],n=e[3],s=e[4],u=e[5];if(s&&u)throw new Error("Attention cannot have both past and attention_bias");if(r.dims.length!==3)throw new Error('Input "input" must have 3 dimensions');let d=r.dims[0],p=r.dims[1],c=r.dims[2];if(a.dims.length!==1)throw new Error('Input "bias" is expected to have 1 dimensions');if(i.dims.length!==2)throw new Error('Input "weights" is expected to have 2 dimensions');if(i.dims[0]!==c)throw new Error("Input 1 dimension 0 should have same length as dimension 2 of input 0");if(a.dims[0]!==i.dims[1])throw new Error('Input "bias" dimension 0 should have same length as dimension 1 of input "weights"');let f=a.dims[0]/3,g=f,_=g;if(t.qkvHiddenSizes.length>0){if(t.qkvHiddenSizes.length!==3)throw new Error("qkv_hidden_sizes attribute should have 3 elements");for(let I of t.qkvHiddenSizes)if(I%t.numHeads!==0)throw new Error("qkv_hidden_sizes should be divisible by num_heads");f=t.qkvHiddenSizes[0],g=t.qkvHiddenSizes[1],_=t.qkvHiddenSizes[2]}let w=p;if(f!==g)throw new Error("qkv_hidden_sizes first element should be same as the second");if(a.dims[0]!==f+g+_)throw new Error('Input "bias" dimension 0 should have same length as sum of Q/K/V hidden sizes');let b=0;if(s){if(g!==_)throw new Error('Input "past" expect k_hidden_size == v_hidden_size');if(s.dims.length!==5)throw new Error('Input "past" must have 5 dimensions');if(s.dims[0]!==2)throw new Error('Input "past" first dimension must be 2');if(s.dims[1]!==d)throw new Error('Input "past" second dimension must be batch_size');if(s.dims[2]!==t.numHeads)throw new Error('Input "past" third dimension must be num_heads');if(s.dims[4]!==g/t.numHeads)throw new Error('Input "past" fifth dimension must be k_hidden_size / num_heads');t.pastPresentShareBuffer||(b=s.dims[3])}let S=w+b,v=-1,$=0;if(n)throw new Error("Mask not supported");if(s)throw new Error("past is not supported");if(u){if(u.dims.length!==4)throw new Error('Input "attention_bias" must have 4 dimensions');if(u.dims[0]!==d||u.dims[1]!==t.numHeads||u.dims[2]!==p||u.dims[3]!==S)throw new Error('Expect "attention_bias" shape (batch_size, num_heads, sequence_length, total_sequence_length)')}return{batchSize:d,sequenceLength:p,pastSequenceLength:b,kvSequenceLength:w,totalSequenceLength:S,maxSequenceLength:v,inputHiddenSize:c,hiddenSize:f,vHiddenSize:_,headSize:Math.floor(f/t.numHeads),vHeadSize:Math.floor(_/t.numHeads),numHeads:t.numHeads,isUnidirectional:!1,pastPresentShareBuffer:!1,maskFilterValue:t.maskFilterValue,maskType:$,scale:t.scale,broadcastResPosBias:!1,passPastInKv:!1,qkvFormat:1}},zr=(e,t,r)=>t&&e?`
      let total_sequence_length_input = u32(${t.getByOffset("0")});
      let present_sequence_length = max(total_sequence_length_input, uniforms.past_sequence_length);
      let is_subsequent_prompt: bool = sequence_length > 1 && sequence_length != total_sequence_length_input;
      let is_first_prompt: bool = is_subsequent_prompt == false && sequence_length == total_sequence_length_input;
      total_sequence_length = u32(${e==null?void 0:e.getByOffset("batchIdx")}) + 1;
      var past_sequence_length: u32 = 0;
      if (is_first_prompt == false) {
        past_sequence_length = total_sequence_length - sequence_length;
      }
       `:`
    ${r?"let past_sequence_length = uniforms.past_sequence_length":""};
    let present_sequence_length = total_sequence_length;
    `,iu=(e,t,r,i,a,n,s,u)=>{let d=$e(s?1:n),p=64,c=n/d;c<p&&(p=32);let f=Math.ceil(n/d/p),g=[{type:12,data:t},{type:12,data:r},{type:12,data:i},{type:12,data:a},{type:12,data:c},{type:12,data:f}],_=ke(e.dataType,d),w=Ce(1,d),b=["type"];s&&b.push("type"),u&&b.push("type");let S=v=>{let $=j("x",e.dataType,e.dims,d),I=[$],T=s?M("seq_lens",s.dataType,s.dims):void 0;T&&I.push(T);let E=u?M("total_sequence_length_input",u.dataType,u.dims):void 0;E&&I.push(E);let A=Ce(e.dataType),C=[{name:"batch_size",type:"u32"},{name:"num_heads",type:"u32"},{name:"past_sequence_length",type:"u32"},{name:"sequence_length",type:"u32"},{name:"total_sequence_length",type:"u32"},{name:"elements_per_thread",type:"u32"}];return`
  var<workgroup> thread_max: array<f32, ${p}>;
  var<workgroup> thread_sum: array<f32, ${p}>;
  ${v.registerUniforms(C).declareVariables(...I)}
  ${v.mainStart([p,1,1])}
    let batchIdx = workgroup_id.z / uniforms.num_heads;
    let headIdx = workgroup_id.z % uniforms.num_heads;
    let sequence_length = uniforms.sequence_length;
    var total_sequence_length = uniforms.total_sequence_length;
    ${zr(T,E,!1)}
    let local_offset = local_idx * uniforms.elements_per_thread;
    let offset = (global_idx / ${p}) * uniforms.total_sequence_length + local_offset;
    let seq_causal_length = ${s?"u32(past_sequence_length + workgroup_id.y + 1)":"total_sequence_length"};
    var thread_max_vector = ${w}(-3.402823e+38f);
    for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {
      thread_max_vector = max(${w}(x[offset + i]), thread_max_vector);
    }
    thread_max[local_idx] = ${(()=>{switch(d){case 1:return"thread_max_vector";case 2:return"max(thread_max_vector.x, thread_max_vector.y)";case 4:return"max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))";default:throw new Error(`Unsupported components: ${d}`)}})()};
    workgroupBarrier();

    var max_value =  f32(-3.402823e+38f);
    for (var i = 0u; i < ${p}; i++) {
      max_value = max(thread_max[i], max_value);
    }

    var sum_vector = ${w}(0);
    for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {
      sum_vector += exp(${w}(x[offset + i]) - max_value);
    }
    thread_sum[local_idx] = ${(()=>{switch(d){case 1:return"sum_vector";case 2:return"sum_vector.x + sum_vector.y";case 4:return"sum_vector.x + sum_vector.y + sum_vector.z + sum_vector.w";default:throw new Error(`Unsupported components: ${d}`)}})()};
    workgroupBarrier();

    var sum: f32 = 0;
    for (var i = 0u; i < ${p}; i++) {
      sum += thread_sum[i];
    }

    if (sum == 0) {
      for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {
        x[offset + i] = ${$.type.value}(${A}(1.0) / ${A}(seq_causal_length));
      }
    } else {
      for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {
        var f32input = ${w}(x[offset + i]);
        x[offset + i] = ${$.type.value}(exp(f32input - max_value) / sum);
      }
    }
      ${s?`
        for (var total_seq_id: u32 = seq_causal_length; total_seq_id + local_offset < uniforms.total_sequence_length; total_seq_id++) {
          x[offset + total_seq_id] = ${$.type.value}(${A}(0));
        }`:""};
  }`};return{name:"AttentionProbsSoftmax",shaderCache:{hint:`${p};${_};${d}`,inputDependencies:b},getShaderSource:S,getRunData:()=>({outputs:[],dispatchGroup:{x:1,y:a,z:t*r},programUniforms:g})}},au=(e,t,r,i,a,n,s,u,d)=>{let p=s+n.kvSequenceLength,c=[n.batchSize,n.numHeads,n.sequenceLength,p],f=e>1&&i,g=n.kvNumHeads?n.kvNumHeads:n.numHeads,_=f?[n.batchSize,g,p,n.headSize]:void 0,w=n.nReps?n.nReps:1,b=n.scale===0?1/Math.sqrt(n.headSize):n.scale,S=$e(n.headSize),v=n.headSize/S,$=12,I={x:Math.ceil(p/$),y:Math.ceil(n.sequenceLength/$),z:n.batchSize*n.numHeads},T=[{type:12,data:n.sequenceLength},{type:12,data:v},{type:12,data:p},{type:12,data:n.numHeads},{type:12,data:n.headSize},{type:1,data:b},{type:12,data:s},{type:12,data:n.kvSequenceLength},{type:12,data:w}],E=f&&i&&B.size(i.dims)>0,A=["type","type"];E&&A.push("type"),a&&A.push("type"),u&&A.push("type"),d&&A.push("type");let C=[{dims:c,dataType:t.dataType,gpuDataType:0}];f&&C.push({dims:_,dataType:t.dataType,gpuDataType:0});let O=U=>{let x=M("q",t.dataType,t.dims,S),Y=M("key",r.dataType,r.dims,S),G=[x,Y];if(E){let X=M("past_key",i.dataType,i.dims,S);G.push(X)}a&&G.push(M("attention_bias",a.dataType,a.dims));let V=u?M("seq_lens",u.dataType,u.dims):void 0;V&&G.push(V);let te=d?M("total_sequence_length_input",d.dataType,d.dims):void 0;te&&G.push(te);let ee=j("output",t.dataType,c),F=[ee];f&&F.push(j("present_key",t.dataType,_,S));let R=Ce(1,S),q=[{name:"M",type:"u32"},{name:"K",type:"u32"},{name:"N",type:"u32"},{name:"num_heads",type:"u32"},{name:"head_size",type:"u32"},{name:"alpha",type:"f32"},{name:"past_sequence_length",type:"u32"},{name:"kv_sequence_length",type:"u32"},{name:"n_reps",type:"u32"}];return`
  const TILE_SIZE = ${$}u;

  var<workgroup> tileQ: array<${x.type.storage}, ${$*$}>;
  var<workgroup> tileK: array<${x.type.storage}, ${$*$}>;
  ${U.registerUniforms(q).declareVariables(...G,...F)}
  ${U.mainStart([$,$,1])}
    // x holds the N and y holds the M
    let headIdx = workgroup_id.z % uniforms.num_heads;
    let kvHeadIdx = ${w===1?"headIdx":"headIdx / uniforms.n_reps"};
    let kv_num_heads = ${w===1?"uniforms.num_heads":"uniforms.num_heads / uniforms.n_reps"};
    let batchIdx = workgroup_id.z / uniforms.num_heads;
    let m = workgroup_id.y * TILE_SIZE;
    let n = workgroup_id.x * TILE_SIZE;
    let sequence_length = uniforms.M;
    var total_sequence_length = uniforms.N;
    ${zr(V,te,!0)}
    let absKvHeadIdx = batchIdx * kv_num_heads + kvHeadIdx;
    let qOffset = workgroup_id.z * uniforms.M * uniforms.K + m * uniforms.K;
    ${E&&f?"let pastKeyOffset = absKvHeadIdx * uniforms.past_sequence_length * uniforms.K;":""};
    let kOffset = absKvHeadIdx * uniforms.kv_sequence_length * uniforms.K;
    ${f?"let presentKeyOffset = absKvHeadIdx * uniforms.N * uniforms.K;":""}
    var value = ${R}(0);
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (global_id.y < uniforms.M && w + local_id.x < uniforms.K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];
      }
      if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {
        var idx = TILE_SIZE * local_id.y + local_id.x;
      ${E&&f?`
              if (n + local_id.y < past_sequence_length) {
                tileK[idx] = past_key[pastKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x];
              } else if (n + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {
                tileK[idx] = key[kOffset + (n + local_id.y - past_sequence_length) * uniforms.K + w + local_id.x];
              }`:`
          if (n + local_id.y < uniforms.kv_sequence_length) {
            tileK[idx] = key[kOffset + (n + local_id.y) * uniforms.K + w + local_id.x];
          }`}
      ${f?`if (n + local_id.y < present_sequence_length) {
        present_key[presentKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x] = tileK[idx];
      }`:""}
      }
      workgroupBarrier();

      for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {
          value += ${R}(tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k]);
      }

      workgroupBarrier();
    }

    if (global_id.y < uniforms.M && global_id.x < total_sequence_length) {
      let headOffset = workgroup_id.z * uniforms.M * uniforms.N;
      let outputIdx = headOffset + global_id.y * uniforms.N + global_id.x;
      var sum: f32 = ${(()=>{switch(S){case 1:return"value";case 2:return"value.x + value.y";case 4:return"value.x + value.y + value.z + value.w";default:throw new Error(`Unsupported components: ${S}`)}})()};
        output[outputIdx] = ${ee.type.value} (sum * uniforms.alpha) + ${a?"attention_bias[outputIdx]":"0.0"};
    }
  }`};return{name:"AttentionProbs",shaderCache:{hint:`${S};${a!==void 0};${i!==void 0};${e}`,inputDependencies:A},getRunData:()=>({outputs:C,dispatchGroup:I,programUniforms:T}),getShaderSource:O}},nu=(e,t,r,i,a,n,s=void 0,u=void 0)=>{let d=n+a.kvSequenceLength,p=a.nReps?a.nReps:1,c=a.vHiddenSize*p,f=e>1&&i,g=a.kvNumHeads?a.kvNumHeads:a.numHeads,_=f?[a.batchSize,g,d,a.headSize]:void 0,w=[a.batchSize,a.sequenceLength,c],b=12,S={x:Math.ceil(a.vHeadSize/b),y:Math.ceil(a.sequenceLength/b),z:a.batchSize*a.numHeads},v=[{type:12,data:a.sequenceLength},{type:12,data:d},{type:12,data:a.vHeadSize},{type:12,data:a.numHeads},{type:12,data:a.headSize},{type:12,data:c},{type:12,data:n},{type:12,data:a.kvSequenceLength},{type:12,data:p}],$=f&&i&&B.size(i.dims)>0,I=["type","type"];$&&I.push("type"),s&&I.push("type"),u&&I.push("type");let T=[{dims:w,dataType:t.dataType,gpuDataType:0}];f&&T.push({dims:_,dataType:t.dataType,gpuDataType:0});let E=A=>{let C=M("probs",t.dataType,t.dims),O=M("v",r.dataType,r.dims),U=[C,O];$&&U.push(M("past_value",i.dataType,i.dims));let x=s?M("seq_lens",s.dataType,s.dims):void 0;s&&U.push(x);let Y=u?M("total_sequence_length_input",u.dataType,u.dims):void 0;u&&U.push(Y);let G=[j("output",t.dataType,w)];f&&G.push(j("present_value",t.dataType,_));let V=[{name:"M",type:"u32"},{name:"K",type:"u32"},{name:"N",type:"u32"},{name:"num_heads",type:"u32"},{name:"head_size",type:"u32"},{name:"v_hidden_size",type:"u32"},{name:"past_sequence_length",type:"u32"},{name:"kv_sequence_length",type:"u32"},{name:"n_reps",type:"u32"}];return`
  const TILE_SIZE = ${b}u;
  var<workgroup> tileQ: array<${C.type.value}, ${b*b}>;
  var<workgroup> tileV: array<${C.type.value}, ${b*b}>;
  ${A.registerUniforms(V).declareVariables(...U,...G)}
  ${A.mainStart([b,b,1])}
   let headIdx = workgroup_id.z % uniforms.num_heads;
   let batchIdx = workgroup_id.z / uniforms.num_heads;
   let kvHeadIdx = ${p===1?"headIdx":"headIdx / uniforms.n_reps"};
   let kv_num_heads = ${p===1?"uniforms.num_heads":"uniforms.num_heads / uniforms.n_reps"};
   let m = global_id.y;
   let n = global_id.x;
   let sequence_length = uniforms.M;
   var total_sequence_length = uniforms.K;
   ${zr(x,Y,!0)}
   let offsetA = workgroup_id.z * uniforms.M * uniforms.K + m * uniforms.K;
   let absKvHeadIdx = batchIdx * kv_num_heads + kvHeadIdx; // kvHeadIdx is relative to the batch
   ${$&&f?"let pastValueOffset = absKvHeadIdx * uniforms.N * uniforms.past_sequence_length + n;":""};
   let vOffset = absKvHeadIdx * uniforms.N * uniforms.kv_sequence_length + n;
   ${f?"let presentValueOffset = absKvHeadIdx * uniforms.N * uniforms.K + n;":""}
   var value = ${C.type.storage}(0);
   for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m < uniforms.M && w + local_id.x < uniforms.K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];
      }
      if (n < uniforms.N && w + local_id.y < uniforms.K) {
        var idx = TILE_SIZE * local_id.y + local_id.x;
        ${$&&f?`
        if (w + local_id.y < past_sequence_length) {
          tileV[idx] = past_value[pastValueOffset + (w + local_id.y) * uniforms.N];
        } else if (w + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {
          tileV[idx] = v[vOffset + (w + local_id.y - past_sequence_length) * uniforms.N];
        }
      `:`
            if (w + local_id.y < uniforms.kv_sequence_length) {
              tileV[idx] = v[vOffset + (w + local_id.y) * uniforms.N];
            }`}
        ${f?`
            if (w + local_id.y < present_sequence_length) {
          present_value[presentValueOffset + (w + local_id.y) * uniforms.N] = tileV[idx];
        }`:""}
      }
     workgroupBarrier();
     for (var k: u32 = 0u; k < TILE_SIZE && w+k < total_sequence_length; k++) {
       value += tileQ[TILE_SIZE * local_id.y + k] * tileV[TILE_SIZE * k + local_id.x];
     }
     workgroupBarrier();
   }

   // we need to transpose output from BNSH_v to BSND_v
   if (m < uniforms.M && n < uniforms.N) {
     let outputIdx = batchIdx * uniforms.M * uniforms.v_hidden_size + m * uniforms.v_hidden_size
       + headIdx * uniforms.N + n;
     output[outputIdx] = value;
   }
  }`};return{name:"AttentionScore",shaderCache:{hint:`${i!==void 0};${e}`,inputDependencies:I},getRunData:()=>({outputs:T,dispatchGroup:S,programUniforms:v}),getShaderSource:E}},pr=(e,t,r,i,a,n,s,u,d,p,c=void 0,f=void 0)=>{let g=Math.min(e.outputCount,1+(s?1:0)+(u?1:0)),_=g>1?p.pastSequenceLength:0,w=_+p.kvSequenceLength,b=d&&B.size(d.dims)>0?d:void 0,S=[t,r];g>1&&s&&B.size(s.dims)>0&&S.push(s),b&&S.push(b),c&&S.push(c),f&&S.push(f);let v=e.compute(au(g,t,r,s,b,p,_,c,f),{inputs:S,outputs:g>1?[-1,1]:[-1]})[0];e.compute(iu(v,p.batchSize,p.numHeads,_,p.sequenceLength,w,c,f),{inputs:c&&f?[v,c,f]:[v],outputs:[]});let $=[v,i];g>1&&u&&B.size(u.dims)>0&&$.push(u),c&&$.push(c),f&&$.push(f),e.compute(nu(g,v,i,u,p,_,c,f),{inputs:$,outputs:g>1?[0,2]:[0]})},su=(e,t)=>{let r=[t.batchSize,t.numHeads,t.sequenceLength,t.headSize],i=t.sequenceLength,a=t.inputHiddenSize,n=t.headSize,s=12,u={x:Math.ceil(t.headSize/s),y:Math.ceil(t.sequenceLength/s),z:t.batchSize*t.numHeads},d=[e.inputs[0],e.inputs[1],e.inputs[2]],p=[{type:12,data:i},{type:12,data:a},{type:12,data:n},{type:12,data:t.numHeads},{type:12,data:t.headSize},{type:12,data:t.hiddenSize},{type:12,data:t.hiddenSize+t.hiddenSize+t.vHiddenSize}],c=f=>{let g=j("output_q",d[0].dataType,r),_=j("output_k",d[0].dataType,r),w=j("output_v",d[0].dataType,r),b=M("input",d[0].dataType,d[0].dims),S=M("weight",d[1].dataType,d[1].dims),v=M("bias",d[2].dataType,d[2].dims),$=b.type.storage,I=[{name:"M",type:"u32"},{name:"K",type:"u32"},{name:"N",type:"u32"},{name:"num_heads",type:"u32"},{name:"head_size",type:"u32"},{name:"hidden_size",type:"u32"},{name:"ldb",type:"u32"}];return`
  const TILE_SIZE = ${s}u;
  var<workgroup> tileInput: array<${$}, ${s*s}>;
  var<workgroup> tileWeightQ: array<${$}, ${s*s}>;
  var<workgroup> tileWeightK: array<${$}, ${s*s}>;
  var<workgroup> tileWeightV: array<${$}, ${s*s}>;
  ${f.registerUniforms(I).declareVariables(b,S,v,g,_,w)}
  ${f.mainStart([s,s,1])}
    let batchIndex = workgroup_id.z / uniforms.num_heads;
    let headNumber = workgroup_id.z % uniforms.num_heads;
    let m = global_id.y;
    let n = global_id.x;

    let inputOffset = batchIndex * (uniforms.M * uniforms.K) + m * uniforms.K;
    let biasOffsetQ = headNumber * uniforms.head_size;
    let biasOffsetK = uniforms.hidden_size + biasOffsetQ;
    let biasOffsetV = uniforms.hidden_size + biasOffsetK;

    var valueQ = ${$}(0);
    var valueK = ${$}(0);
    var valueV = ${$}(0);
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m < uniforms.M && w + local_id.x < uniforms.K) {
        tileInput[TILE_SIZE * local_id.y + local_id.x] = input[inputOffset + w + local_id.x];
      }
      if (n < uniforms.N && w + local_id.y < uniforms.K) {
        let offset = n + (w + local_id.y) * uniforms.ldb;
        tileWeightQ[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetQ + offset];
        tileWeightK[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetK + offset];
        tileWeightV[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetV + offset];
      }
      workgroupBarrier();
      for (var k: u32 = 0u; k<TILE_SIZE && w+k < uniforms.K; k++) {
        let inputTileOffset = TILE_SIZE * local_id.y + k;
        let weightTileOffset = TILE_SIZE * k + local_id.x;
        valueQ += tileInput[inputTileOffset] * tileWeightQ[weightTileOffset];
        valueK += tileInput[inputTileOffset] * tileWeightK[weightTileOffset];
        valueV += tileInput[inputTileOffset] * tileWeightV[weightTileOffset];
      }

      workgroupBarrier();
    }

    let headOffset = (m * uniforms.N + n) % uniforms.head_size;
    valueQ += bias[headOffset + biasOffsetQ];
    valueK += bias[headOffset + biasOffsetK];
    valueV += bias[headOffset + biasOffsetV];

    let offset = workgroup_id.z * uniforms.M * uniforms.N;
    if (m < uniforms.M && n < uniforms.N) {
      let outputIdx = offset + m * uniforms.N + n;
      output_q[outputIdx] = valueQ;
      output_k[outputIdx] = valueK;
      output_v[outputIdx] = valueV;
    }
  }`};return e.compute({name:"AttentionPrepare",shaderCache:{inputDependencies:["type","type","type"]},getRunData:()=>({outputs:[{dims:r,dataType:e.inputs[0].dataType,gpuDataType:0},{dims:r,dataType:e.inputs[0].dataType,gpuDataType:0},{dims:r,dataType:e.inputs[0].dataType,gpuDataType:0}],dispatchGroup:u,programUniforms:p}),getShaderSource:c},{inputs:d,outputs:[-1,-1,-1]})},Lp=(e,t)=>{let r=ru(e.inputs,t),[i,a,n]=su(e,r);return pr(e,i,a,n,e.inputs[4],void 0,void 0,void 0,e.inputs[5],r)}}),ou,uu,lu,Vp,t0=P(()=>{Ue(),J(),ie(),ve(),ae(),ou=(e,t)=>{if(!e||e.length!==5)throw new Error("BatchNormalization requires 5 inputs");let r=(i,a,n)=>{let s=a.length;if(s!==i.length)throw new Error(`${n}: num dimensions != ${s}`);a.forEach((u,d)=>{if(u!==i[d])throw new Error(`${n}: dim[${d}] do not match`)})};if(e[0].dims.length>1){let i=t.format==="NHWC"?t.spatial?e[0].dims.slice(-1):e[0].dims.slice(-1).concat(e[0].dims.slice(1,e[0].dims.length-1)):e[0].dims.slice(1,t.spatial?2:void 0);r(e[1].dims,i,"Invalid input scale"),r(e[2].dims,i,"Invalid input B"),r(e[3].dims,i,"Invalid input mean"),r(e[4].dims,i,"Invalid input var")}else r(e[1].dims,[1],"Invalid input scale"),r(e[2].dims,[1],"Invalid input B"),r(e[3].dims,[1],"Invalid input mean"),r(e[4].dims,[1],"Invalid input var")},uu=(e,t)=>{let{epsilon:r,spatial:i,format:a}=t,n=e[0].dims,s=i?$e(n[n.length-1]):1,u=a==="NHWC"&&n.length>1?s:1,d=B.size(n)/s,p=i,c=p?n.length:n,f=M("x",e[0].dataType,e[0].dims,s),g=M("scale",e[1].dataType,e[1].dims,u),_=M("bias",e[2].dataType,e[2].dims,u),w=M("inputMean",e[3].dataType,e[3].dims,u),b=M("inputVar",e[4].dataType,e[4].dims,u),S=j("y",e[0].dataType,c,s),v=()=>{let I="";if(i)I=`let cOffset = ${n.length===1?"0u":a==="NHWC"?`outputIndices[${n.length-1}] / ${s}`:"outputIndices[1]"};`;else if(a==="NCHW")I=`
            ${S.indicesSet("outputIndices","0","0")}
            let cOffset = ${S.indicesToOffset("outputIndices")};`;else{I=`var cIndices = ${g.type.indices}(0);
                       cIndices[0] = outputIndices[${n.length-1}];`;for(let T=1;T<g.rank;T++)I+=`cIndices[${T}] = outputIndices[${T}];`;I+=`let cOffset = ${g.indicesToOffset("cIndices")};`}return I},$=I=>`
  const epsilon = ${r};
  ${I.registerUniform("outputSize","u32").declareVariables(f,g,_,w,b,S)}
  ${I.mainStart()}
  ${I.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
    var outputIndices = ${S.offsetToIndices(`global_idx * ${s}`)};
    ${v()}
    let scale = ${g.getByOffset("cOffset")};
    let bias = ${_.getByOffset("cOffset")};
    let inputMean = ${w.getByOffset("cOffset")};
    let inputVar = ${b.getByOffset("cOffset")};
    let x = ${f.getByOffset("global_idx")};
    let value = (x - inputMean) * inverseSqrt(inputVar + epsilon) * scale + bias;
    ${S.setByOffset("global_idx","value")}
  }`;return{name:"BatchNormalization",shaderCache:{hint:`${t.epsilon}_${t.format}_${i}_${s}`,inputDependencies:p?["rank","type","type","type","type"]:void 0},getShaderSource:$,getRunData:()=>({outputs:[{dims:e[0].dims,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(d/64)},programUniforms:p?[{type:12,data:d},...Q(n)]:[{type:12,data:d}]})}},lu=e=>ce(e),Vp=(e,t)=>{let{inputs:r,outputCount:i}=e,a=lu({...t,outputCount:i});if(ye.webgpu.validateInputContent&&ou(r,a),t.trainingMode)throw new Error("BatchNormalization trainingMode is not supported yet.");e.compute(uu(r,a))}}),du,pu,Gp,r0=P(()=>{ie(),ae(),du=e=>{if(e[0].dims.length!==3)throw new Error("input should have 3 dimensions");if(![320,640,1280].includes(e[0].dims[2]))throw new Error("number of channels should be 320, 640 or 1280");if(e[1].dims.length!==1)throw new Error("bias is expected to have 1 dimensions");if(e[0].dims[2]!==e[1].dims[0])throw new Error("last dimension of input and bias are not the same")},pu=e=>{let t=e[0].dims,r=e[0].dims[2],i=B.size(t)/4,a=e[0].dataType,n=M("input",a,t,4),s=M("bias",a,[r],4),u=M("residual",a,t,4),d=j("output",a,t,4);return{name:"BiasAdd",getRunData:()=>({outputs:[{dims:t,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(i/64)}}),getShaderSource:p=>`
  const channels = ${r}u / 4;
  ${p.declareVariables(n,s,u,d)}

  ${p.mainStart()}
    ${p.guardAgainstOutOfBoundsWorkgroupSizes(i)}
    let value = ${n.getByOffset("global_idx")}
      + ${s.getByOffset("global_idx % channels")} + ${u.getByOffset("global_idx")};
    ${d.setByOffset("global_idx","value")}
  }`}},Gp=e=>{du(e.inputs),e.compute(pu(e.inputs))}}),cu,pe,Hp,Fp,jp,Kp,Zp,Qp,Yp,Xp,Jp,hu,ec,tc,rc,ic,or,ac,Mr,nc,sc,oc,uc,lc,dc,pc,cc,hc,fc,mc,gc,yc,_c,wc,bc,Ui,$c,ba,$a,vc,xc,Sc,fu,mu,kc,Ga=P(()=>{J(),ie(),ve(),ae(),cu=(e,t,r,i,a,n,s)=>{let u=Math.ceil(t/4),d="";typeof a=="string"?d=`${a}(a)`:d=a("a");let p=M("inputData",r,[u],4),c=j("outputData",i,[u],4),f=[{name:"vec_size",type:"u32"}];return s&&f.push(...s),`
      ${e.registerUniforms(f).declareVariables(p,c)}

  ${n??""}

  ${e.mainStart()}
    ${e.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")}

    let a = ${p.getByOffset("global_idx")};
    ${c.setByOffset("global_idx",d)}
  }`},pe=(e,t,r,i,a,n=e.dataType,s,u)=>{let d=[{type:12,data:Math.ceil(B.size(e.dims)/4)}];return s&&d.push(...s),{name:t,shaderCache:{hint:a,inputDependencies:["type"]},getShaderSource:p=>cu(p,B.size(e.dims),e.dataType,n,r,i,u),getRunData:p=>({outputs:[{dims:e.dims,dataType:n}],dispatchGroup:{x:Math.ceil(B.size(p[0].dims)/64/4)},programUniforms:d})}},Hp=e=>{e.compute(pe(e.inputs[0],"Abs","abs"))},Fp=e=>{e.compute(pe(e.inputs[0],"Acos","acos"))},jp=e=>{e.compute(pe(e.inputs[0],"Acosh","acosh"))},Kp=e=>{e.compute(pe(e.inputs[0],"Asin","asin"))},Zp=e=>{e.compute(pe(e.inputs[0],"Asinh","asinh"))},Qp=e=>{e.compute(pe(e.inputs[0],"Atan","atan"))},Yp=e=>{e.compute(pe(e.inputs[0],"Atanh","atanh"))},Xp=e=>ce(e),Jp=(e,t)=>{let r;switch(t.to){case 10:r="vec4<f16>";break;case 1:r="vec4<f32>";break;case 12:r="vec4<u32>";break;case 6:r="vec4<i32>";break;case 9:r="vec4<bool>";break;default:throw new RangeError(`not supported type (specified in attribute 'to' from 'Cast' operator): ${t.to}`)}e.compute(pe(e.inputs[0],"Cast",r,void 0,t.cacheKey,t.to))},hu=e=>{let t,r,i=e.length>=2&&e[1].data!==0,a=e.length>=3&&e[2].data!==0;switch(e[0].dataType){case 1:t=i?e[1].getFloat32Array()[0]:-34028234663852886e22,r=a?e[2].getFloat32Array()[0]:34028234663852886e22;break;case 10:t=i?e[1].getUint16Array()[0]:64511,r=a?e[2].getUint16Array()[0]:31743;break;default:throw new Error("Unsupport data type")}return ce({min:t,max:r})},ec=(e,t)=>{let r=t||hu(e.inputs),i=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"Clip",a=>`clamp(${a}, vec4<${i}>(uniforms.min), vec4<${i}>(uniforms.max))`,void 0,r.cacheKey,void 0,[{type:e.inputs[0].dataType,data:r.min},{type:e.inputs[0].dataType,data:r.max}],[{name:"min",type:i},{name:"max",type:i}]),{inputs:[0]})},tc=e=>{e.compute(pe(e.inputs[0],"Ceil","ceil"))},rc=e=>{e.compute(pe(e.inputs[0],"Cos","cos"))},ic=e=>{e.compute(pe(e.inputs[0],"Cosh","cosh"))},or=e=>ce(e),ac=(e,t)=>{let r=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"Elu",i=>`elu_vf32(${i})`,`
  const elu_alpha_ = ${r}(${t.alpha});

  fn elu_f32(a: ${r}) -> ${r} {
  return select((exp(a) - 1.0) * elu_alpha_, a, a >= 0.0);
  }

  fn elu_vf32(v: vec4<${r}>) -> vec4<${r}> {
  return vec4(elu_f32(v.x), elu_f32(v.y), elu_f32(v.z), elu_f32(v.w));
  }`,t.cacheKey))},Mr=(e="f32")=>`
const r0: ${e} = 0.3275911;
const r1: ${e} = 0.254829592;
const r2: ${e} = -0.284496736;
const r3: ${e} = 1.421413741;
const r4: ${e} = -1.453152027;
const r5: ${e} = 1.061405429;

fn erf_vf32(v: vec4<${e}>) -> vec4<${e}> {
  let absv = abs(v);
  let x = 1.0 / (1.0 + r0 * absv);
  return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));
}`,nc=e=>{let t=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"Erf",r=>`erf_vf32(${r})`,Mr(t)))},sc=e=>{e.compute(pe(e.inputs[0],"Exp","exp"))},oc=e=>{e.compute(pe(e.inputs[0],"Floor","floor"))},uc=e=>{let t=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"Gelu",r=>`0.5 * ${r} * (1.0 + erf_vf32(${r} * 0.7071067811865475))`,Mr(t)))},lc=(e,t)=>{let r=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"LeakyRelu",i=>`select(leaky_relu_alpha_ * ${i}, ${i}, ${i} >= vec4<${r}>(0.0))`,`const leaky_relu_alpha_ = ${r}(${t.alpha});`,t.cacheKey))},dc=e=>{e.compute(pe(e.inputs[0],"Not",t=>`!${t}`))},pc=e=>{e.compute(pe(e.inputs[0],"Neg",t=>`-${t}`))},cc=e=>{e.compute(pe(e.inputs[0],"Reciprocal",t=>`1.0/${t}`))},hc=e=>{let t=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"Relu",r=>`select(vec4<${t}>(0.0), ${r}, ${r} > vec4<${t}>(0.0))`))},fc=e=>{e.compute(pe(e.inputs[0],"Sigmoid",t=>`(1.0 / (1.0 + exp(-${t})))`))},mc=e=>ce(e),gc=(e,t)=>{let r=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"HardSigmoid",i=>`max(vec4<${r}>(0.0), min(vec4<${r}>(1.0), ${t.alpha} * ${i} + vec4<${r}>(${t.beta})))`,void 0,t.cacheKey))},yc=e=>{e.compute(pe(e.inputs[0],"Sin","sin"))},_c=e=>{e.compute(pe(e.inputs[0],"Sinh","sinh"))},wc=e=>{e.compute(pe(e.inputs[0],"Sqrt","sqrt"))},bc=e=>{e.compute(pe(e.inputs[0],"Tan","tan"))},Ui=e=>`sign(${e}) * (1 - exp(-2 * abs(${e}))) / (1 + exp(-2 * abs(${e})))`,$c=e=>{e.compute(pe(e.inputs[0],"Tanh",Ui))},ba=(e="f32")=>`
const fast_gelu_a: ${e} = 0.5;
const fast_gelu_b: ${e} = 0.7978845608028654;
const fast_gelu_c: ${e} = 0.035677408136300125;

fn tanh_v(v: vec4<${e}>) -> vec4<${e}> {
  return ${Ui("v")};
}
`,$a=e=>`(fast_gelu_a + fast_gelu_a * tanh_v(${e} * (fast_gelu_c * ${e} * ${e} + fast_gelu_b))) * ${e}`,vc=e=>{let t=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"FastGelu",$a,ba(t),void 0,e.inputs[0].dataType))},xc=(e,t)=>{let r=Ce(e.inputs[0].dataType);return e.compute(pe(e.inputs[0],"ThresholdedRelu",i=>`select(vec4<${r}>(0.0), ${i}, ${i} > thresholded_relu_alpha_)`,`const thresholded_relu_alpha_ = vec4<${r}>(${t.alpha});`,t.cacheKey)),0},Sc=e=>{e.compute(pe(e.inputs[0],"Log","log"))},fu=(e,t)=>`
const alpha = vec4<${e}>(${t});
const one = ${e}(1.0);
const zero = ${e}(0.0);

fn quick_gelu_impl(x: vec4<${e}>) -> vec4<${e}> {
  let v = x *alpha;
  var x1 : vec4<${e}>;
  for (var i = 0; i < 4; i = i + 1) {
    if (v[i] >= zero) {
      x1[i] = one / (one + exp(-v[i]));
    } else {
      x1[i] = one - one / (one + exp(v[i]));
    }
  }
  return x * x1;
}
`,mu=e=>`quick_gelu_impl(${e})`,kc=(e,t)=>{let r=Ce(e.inputs[0].dataType);e.compute(pe(e.inputs[0],"QuickGelu",mu,fu(r,t.alpha),t.cacheKey,e.inputs[0].dataType))}}),gu,yu,Tc,i0=P(()=>{ie(),ae(),Ga(),gu=e=>{if(e[0].dims.length!==3)throw new Error("input should have 3 dimensions");if(![2560,5120,10240].includes(e[0].dims[2]))throw new Error("hidden state should be 2560, 5120 or 10240");if(e[1].dims.length!==1)throw new Error("bias is expected to have 1 dimensions");if(e[0].dims[2]!==e[1].dims[0])throw new Error("last dimension of input and bias are not the same")},yu=e=>{let t=e[0].dims.slice();t[2]=t[2]/2;let r=M("input",e[0].dataType,e[0].dims,4),i=M("bias",e[0].dataType,[e[0].dims[2]],4),a=j("output",e[0].dataType,t,4),n=B.size(t)/4,s=ke(e[0].dataType);return{name:"BiasSplitGelu",getRunData:()=>({outputs:[{dims:t,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(n/64)}}),getShaderSource:u=>`
  const M_SQRT2 = sqrt(2.0);
  const halfChannels = ${e[0].dims[2]/4/2}u;

  ${u.declareVariables(r,i,a)}

  ${Mr(s)}

  ${u.mainStart()}
    ${u.guardAgainstOutOfBoundsWorkgroupSizes(n)}
    let biasIdx = global_idx % halfChannels;
    let batchIndex = global_idx / halfChannels;
    let inputOffset = biasIdx + batchIndex * halfChannels * 2;
    let valueLeft = input[inputOffset] + bias[biasIdx];
    let valueRight = input[inputOffset + halfChannels] + bias[biasIdx + halfChannels];
    let geluRight = valueRight * 0.5 * (erf_vf32(valueRight / M_SQRT2) + 1);

    ${a.setByOffset("global_idx","valueLeft * geluRight")}
  }`}},Tc=e=>{gu(e.inputs),e.compute(yu(e.inputs))}}),_u,wu,He,Ic,Ec,zc,Cc,Ac,Oc,Rc,Bc,Nc,Dc,a0=P(()=>{J(),ie(),ae(),_u=(e,t,r,i,a,n,s,u,d,p,c,f)=>{let g,_;typeof u=="string"?g=_=($,I)=>`${u}((${$}),(${I}))`:typeof u=="function"?g=_=u:(g=u.scalar,_=u.vector);let w=j("outputData",c,i.length,4),b=M("aData",d,t.length,4),S=M("bData",p,r.length,4),v;if(a)if(n){let $=B.size(t)===1,I=B.size(r)===1,T=t.length>0&&t[t.length-1]%4===0,E=r.length>0&&r[r.length-1]%4===0;$||I?v=w.setByOffset("global_idx",_($?`${b.type.value}(${b.getByOffset("0")}.x)`:b.getByOffset("global_idx"),I?`${S.type.value}(${S.getByOffset("0")}.x)`:S.getByOffset("global_idx"))):v=`
            let outputIndices = ${w.offsetToIndices("global_idx * 4u")};
            let offsetA = ${b.broadcastedIndicesToOffset("outputIndices",w)};
            let offsetB = ${S.broadcastedIndicesToOffset("outputIndices",w)};
            ${w.setByOffset("global_idx",_(s||T?b.getByOffset("offsetA / 4u"):`${b.type.value}(${b.getByOffset("offsetA / 4u")}[offsetA % 4u])`,s||E?S.getByOffset("offsetB / 4u"):`${S.type.value}(${S.getByOffset("offsetB / 4u")}[offsetB % 4u])`))}
          `}else v=w.setByOffset("global_idx",_(b.getByOffset("global_idx"),S.getByOffset("global_idx")));else{if(!n)throw new Error("no necessary to use scalar implementation for element-wise binary op implementation.");let $=(I,T,E="")=>{let A=`aData[indexA${T}][componentA${T}]`,C=`bData[indexB${T}][componentB${T}]`;return`
            let outputIndices${T} = ${w.offsetToIndices(`global_idx * 4u + ${T}u`)};
            let offsetA${T} = ${b.broadcastedIndicesToOffset(`outputIndices${T}`,w)};
            let offsetB${T} = ${S.broadcastedIndicesToOffset(`outputIndices${T}`,w)};
            let indexA${T} = offsetA${T} / 4u;
            let indexB${T} = offsetB${T} / 4u;
            let componentA${T} = offsetA${T} % 4u;
            let componentB${T} = offsetB${T} % 4u;
            ${I}[${T}] = ${E}(${g(A,C)});
          `};c===9?v=`
            var data = vec4<u32>(0);
            ${$("data",0,"u32")}
            ${$("data",1,"u32")}
            ${$("data",2,"u32")}
            ${$("data",3,"u32")}
            outputData[global_idx] = dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(data));`:v=`
            ${$("outputData[global_idx]",0)}
            ${$("outputData[global_idx]",1)}
            ${$("outputData[global_idx]",2)}
            ${$("outputData[global_idx]",3)}
          `}return`
        ${e.registerUniform("vec_size","u32").declareVariables(b,S,w)}

        ${f??""}

        ${e.mainStart()}
        ${e.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")}
        ${v}
      }`},wu=(e,t,r,i,a,n,s=r.dataType)=>{let u=r.dims.map(b=>Number(b)??1),d=i.dims.map(b=>Number(b)??1),p=!B.areEqual(u,d),c=u,f=B.size(u),g=!1,_=!1,w=[p];if(p){let b=Wt.calcShape(u,d,!1);if(!b)throw new Error("Can't perform binary op on the given tensors");c=b.slice(),f=B.size(c);let S=B.size(u)===1,v=B.size(d)===1,$=u.length>0&&u[u.length-1]%4===0,I=d.length>0&&d[d.length-1]%4===0;w.push(S),w.push(v),w.push($),w.push(I);let T=1;for(let E=1;E<c.length;E++){let A=u[u.length-E],C=d[d.length-E];if(A===C)T*=A;else break}T%4===0?(_=!0,g=!0):(S||v||$||I)&&(g=!0)}else g=!0;return w.push(g),{name:e,shaderCache:{hint:t+w.map(b=>b.toString()).join("_"),inputDependencies:["rank","rank"]},getShaderSource:b=>_u(b,u,d,c,g,p,_,a,r.dataType,i.dataType,s,n),getRunData:()=>({outputs:[{dims:c,dataType:s}],dispatchGroup:{x:Math.ceil(f/64/4)},programUniforms:[{type:12,data:Math.ceil(B.size(c)/4)},...Q(u,d,c)]})}},He=(e,t,r,i,a,n)=>{e.compute(wu(t,a??"",e.inputs[0],e.inputs[1],r,i,n))},Ic=e=>{He(e,"Add",(t,r)=>`${t}+${r}`)},Ec=e=>{He(e,"Div",(t,r)=>`${t}/${r}`)},zc=e=>{He(e,"Equal",{scalar:(t,r)=>`u32(${t}==${r})`,vector:(t,r)=>`vec4<u32>(${t}==${r})`},void 0,void 0,9)},Cc=e=>{He(e,"Mul",(t,r)=>`${t}*${r}`)},Ac=e=>{let t=M("input",e.inputs[0].dataType,e.inputs[0].dims).type.value;He(e,"Pow",{scalar:(r,i)=>`pow_custom(${r},${i})`,vector:(r,i)=>`pow_vector_custom(${r},${i})`},`
    fn pow_custom(a : ${t}, b : ${t}) -> ${t} {
      if (b == ${t}(0.0)) {
        return ${t}(1.0);
      } else if (a < ${t}(0.0) && f32(b) != floor(f32(b))) {
        return ${t}(pow(f32(a), f32(b))); // NaN
      }
      return select(sign(a), ${t}(1.0), round(f32(abs(b) % ${t}(2.0))) != 1.0) * ${t}(${t==="i32"?"round":""}(pow(f32(abs(a)), f32(b))));
    }
    fn pow_vector_custom(a : vec4<${t}>, b : vec4<${t}>) -> vec4<${t}> {
      // TODO: implement vectorized pow
      return vec4<${t}>(pow_custom(a.x, b.x), pow_custom(a.y, b.y), pow_custom(a.z, b.z), pow_custom(a.w, b.w));
    }
      `)},Oc=e=>{He(e,"Sub",(t,r)=>`${t}-${r}`)},Rc=e=>{He(e,"Greater",{scalar:(t,r)=>`u32(${t}>${r})`,vector:(t,r)=>`vec4<u32>(${t}>${r})`},void 0,void 0,9)},Bc=e=>{He(e,"Less",{scalar:(t,r)=>`u32(${t}<${r})`,vector:(t,r)=>`vec4<u32>(${t}<${r})`},void 0,void 0,9)},Nc=e=>{He(e,"GreaterOrEqual",{scalar:(t,r)=>`u32(${t}>=${r})`,vector:(t,r)=>`vec4<u32>(${t}>=${r})`},void 0,void 0,9)},Dc=e=>{He(e,"LessOrEqual",{scalar:(t,r)=>`u32(${t}<=${r})`,vector:(t,r)=>`vec4<u32>(${t}<=${r})`},void 0,void 0,9)}}),bu,$u,vu,xu,Mc,Uc,n0=P(()=>{J(),ie(),ve(),ae(),bu=(e,t)=>{if(!e||e.length<1)throw new Error("too few inputs");let r=0,i=e[r],a=i.dataType,n=i.dims.length;e.forEach((s,u)=>{if(u!==r){if(s.dataType!==a)throw new Error("input tensors should be one type");if(s.dims.length!==n)throw new Error("input tensors should have the same shape");s.dims.forEach((d,p)=>{if(p!==t&&d!==i.dims[p])throw new Error("non concat dimensions must match")})}})},$u=(e,t)=>`
  fn calculateInputIndex(index: u32) -> u32 {
    let sizeInConcatAxis = array<u32, ${e}u>(${t});
    for (var i: u32 = 0u; i < ${e}; i += 1u ) {
      if (index < sizeInConcatAxis[i]) {
        return i;
      }
    }
    return ${e}u;
  }`,vu=(e,t)=>{let r=e.length,i=[];for(let a=0;a<r;++a){let n=t.setByOffset("global_idx",e[a].getByIndices("indices"));r===1?i.push(n):a===0?i.push(`if (inputIndex == ${a}u) { ${n} }`):a===r-1?i.push(`else { ${n} }`):i.push(`else if (inputIndex == ${a}) { ${n} }`)}return i.join(`
`)},xu=(e,t,r,i)=>{let a=B.size(r),n=new Array(e.length),s=new Array(e.length),u=0,d=[],p=[],c=[{type:12,data:a}];for(let b=0;b<e.length;++b)u+=e[b].dims[t],n[b]=u,p.push(e[b].dims.length),s[b]=M(`input${b}`,i,p[b]),d.push("rank"),c.push({type:12,data:n[b]});for(let b=0;b<e.length;++b)c.push(...Q(e[b].dims));c.push(...Q(r));let f=j("output",i,r.length),g=f.indicesGet("indices",t),_=Array.from(Array(n.length).keys()).map(b=>`uniforms.sizeInConcatAxis${b}`).join(","),w=b=>`

  ${(()=>{b.registerUniform("outputSize","u32");for(let S=0;S<e.length;S++)b.registerUniform(`sizeInConcatAxis${S}`,"u32");return b.declareVariables(...s,f)})()}

  ${$u(n.length,_)}

  ${b.mainStart()}
    ${b.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}

    var indices = ${f.offsetToIndices("global_idx")};

    let inputIndex = calculateInputIndex(${g});
    if (inputIndex != 0u) {
      let sizeInConcatAxis = array<u32, ${n.length}u>(${_});
      ${g} -= sizeInConcatAxis[inputIndex - 1u];
    }

    ${vu(s,f)}
  }`;return{name:"Concat",shaderCache:{hint:`${t}`,inputDependencies:d},getRunData:()=>({outputs:[{dims:r,dataType:i}],dispatchGroup:{x:Math.ceil(a/64)},programUniforms:c}),getShaderSource:w}},Mc=(e,t)=>{let r=e.inputs,i=r[0].dims,a=B.normalizeAxis(t.axis,i.length);bu(r,a);let n=i.slice();n[a]=r.reduce((u,d)=>u+(d.dims.length>a?d.dims[a]:0),0);let s=r.filter(u=>B.size(u.dims)>0);e.compute(xu(s,a,n,r[0].dataType),{inputs:s})},Uc=e=>ce({axis:e.axis})}),At,Ot,Rt,Ha,Nt=P(()=>{J(),ie(),At=(e,t,r="f32")=>{switch(e.activation){case"Relu":return`value = max(value, ${t}(0.0));`;case"Sigmoid":return`value = (${t}(1.0) / (${t}(1.0) + exp(-value)));`;case"Clip":return`value = clamp(value, ${t}(${r}(uniforms.clip_min)), ${t}(${r}(uniforms.clip_max)));`;case"HardSigmoid":return`value = max(${t}(0.0), min(${t}(1.0), ${r}(uniforms.alpha) * value + ${r}(uniforms.beta)));`;case"LeakyRelu":return`value = select(${r}(uniforms.alpha) * value, value, value >= ${t}(0.0));`;case"Tanh":return`let e2x = exp(-2.0 * abs(value));
              value = sign(value) * (1.0 - e2x) / (1.0 + e2x);
        `;case"":return"";default:throw new Error(`Unsupported activation ${e.activation}`)}},Ot=(e,t)=>{e.activation==="Clip"?t.push({type:1,data:e.clipMax},{type:1,data:e.clipMin}):e.activation==="HardSigmoid"?t.push({type:1,data:e.alpha},{type:1,data:e.beta}):e.activation==="LeakyRelu"&&t.push({type:1,data:e.alpha})},Rt=(e,t)=>{e.activation==="Clip"?t.push({name:"clip_max",type:"f32"},{name:"clip_min",type:"f32"}):e.activation==="HardSigmoid"?t.push({name:"alpha",type:"f32"},{name:"beta",type:"f32"}):e.activation==="LeakyRelu"&&t.push({name:"alpha",type:"f32"})},Ha=e=>{let t=(e==null?void 0:e.activation)||"";if(t==="HardSigmoid"){let[r,i]=(e==null?void 0:e.activation_params)||[.2,.5];return{activation:t,alpha:r,beta:i}}else if(t==="Clip"){let[r,i]=(e==null?void 0:e.activation_params)||[dp,pp];return{activation:t,clipMax:i,clipMin:r}}else if(t==="LeakyRelu"){let[r]=(e==null?void 0:e.activation_params)||[.01];return{activation:t,alpha:r}}return{activation:t}}}),Ee,Pc,Fa=P(()=>{Ee=(e,t)=>{switch(e){case 1:return t;case 2:return`vec2<${t}>`;case 3:return`vec3<${t}>`;case 4:return`vec4<${t}>`;default:throw new Error(`${e}-component is not supported.`)}},Pc=e=>`
      ${e?"value = value + getBiasByOutputCoords(coords);":""}
      `}),qc,s0=P(()=>{qc=e=>`
fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
}
fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
  return dot(coords, vec4<i32>(
    i32(${e}.x), i32(${e}.y), i32(${e}.z), 1));
}
`}),lr,ja,Ka=P(()=>{J(),ie(),ae(),Nt(),lr=(e,t,r,i,a)=>{let n=i-r;return`
      ${Array.from({length:r}).map((s,u)=>`
      if (${Z(t.shape,u,t.rank)} != 1) {
        ${t.indicesSet(e,u,Z(a,u+n,i))}
      } else {
        ${t.indicesSet(e,u,0)}
      }`).join("")}
`},ja=(e,t,r,i,a=!1,n)=>{let s=e[0].dims,u=e[1].dims,d=s[s.length-2],p=u[u.length-1],c=s[s.length-1],f=$e(p),g=$e(c),_=$e(d),w=B.size(r)/f/_,b=e.length>2,S=i?i.slice(0,-2):r.slice(0,-2),v=[B.size(S),d,p],$=[{type:12,data:w},{type:12,data:d},{type:12,data:p},{type:12,data:c}];Ot(t,$),$.push(...Q(S,s,u)),b&&$.push(...Q(e[2].dims)),$.push(...Q(v));let I=T=>{let E=Wa("batch_dims",e[0].dataType,S.length),A=M("a",e[0].dataType,s.length,g),C=M("b",e[1].dataType,u.length,f),O=j("output",e[0].dataType,v.length,f),U=ke(O.type.tensor),x=At(t,O.type.value,U),Y=[A,C],G="";if(b){let ee=a?f:1;Y.push(M("bias",e[2].dataType,e[2].dims.length,ee)),G=`${a?`value += bias[col / ${ee}];`:`value += ${O.type.value}(bias[row + i]);`}`}let V=[{name:"output_size",type:"u32"},{name:"M",type:"u32"},{name:"N",type:"u32"},{name:"K",type:"u32"}];Rt(t,V);let te=()=>{let ee=`var a_data: ${A.type.value};`;for(let F=0;F<g;F++)ee+=`
              let b_data${F} = b[(b_offset + (k + ${F}) * uniforms.N + col) / ${f}];`;for(let F=0;F<_;F++){ee+=`a_data = a[(a_offset + (row + ${F}) * uniforms.K + k) / ${g}];`;for(let R=0;R<g;R++)ee+=`
            values[${F}] = fma(${C.type.value}(a_data${g===1?"":`[${R}]`}), b_data${R}, values[${F}]);
`}return ee};return`
  ${T.registerUniforms(V).registerInternalVariables(E).declareVariables(...Y,O)}
  ${T.mainStart()}
    ${T.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    let col = (global_idx % (uniforms.N / ${f})) * ${f};
    var index1 = global_idx / (uniforms.N / ${f});
    let stride1 = uniforms.M / ${_};
    let row = (index1 % stride1) * ${_};
    let batch = index1 / stride1;

    ${r.length===2?"":`let batch_indices = ${E.offsetToIndices("batch")};`}

    var a_indices: ${A.type.indices};
    ${lr("a_indices",A,A.rank-2,E.rank,"batch_indices")}
    ${A.indicesSet("a_indices",A.rank-2,0)}
    ${A.indicesSet("a_indices",A.rank-1,0)}
    let a_offset = ${A.indicesToOffset("a_indices")};

    var b_indices: ${C.type.indices};
    ${lr("b_indices",C,C.rank-2,E.rank,"batch_indices")}
    ${C.indicesSet("b_indices",C.rank-2,0)}
    ${C.indicesSet("b_indices",C.rank-1,0)}
    let b_offset = ${C.indicesToOffset("b_indices")};
    var values: array<${O.type.value}, ${_}>;
    for (var k: u32 = 0u; k < uniforms.K; k = k + ${g}) {
      ${te()}
    }
    for (var i = 0u; i < ${_}u; i++) {
      var value = values[i];
      ${G}
      ${x}
      let cur_indices = ${O.type.indices}(batch, row + i, col);
      let offset = ${O.indicesToOffset("cur_indices")};
      ${O.setByOffset(`offset / ${f}`,"value")};
    }
  }
  `};return{name:"MatMulNaive",shaderCache:{hint:`${t.activation};${f};${g};${_};${a}`,inputDependencies:b?["rank","rank","rank"]:["rank","rank"]},getRunData:()=>({outputs:[{dims:n?n(r):r,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(w/64)},programUniforms:$}),getShaderSource:I}}}),Su,ku,va,Pi,Tu,xa,Iu,Vr,Za=P(()=>{J(),ie(),ae(),Nt(),Ka(),Fa(),Su=(e,t)=>e?`
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          kStart + inputRow,
          globalRowStart / innerElementSize + inputCol${t?", batchIndices":""});
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          globalRow + innerRow,
          kStart / innerElementSize + inputCol${t?", batchIndices":""});
        `,ku=(e,t)=>e?`
        let ACached0 = mm_Asub[k * innerElementSize][localRow];
        let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];
        let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];
        ${t===3?"":"let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];"}
        for (var i = 0; i < rowPerThread; i = i + 1) {
          acc[i] = BCached0 * ACached0[i] + acc[i];
          acc[i] = BCached1 * ACached1[i] + acc[i];
          acc[i] = BCached2 * ACached2[i] + acc[i];
          ${t===3?"":"acc[i] = BCached3 * ACached3[i] + acc[i];"}
        }`:`
        for (var i = 0; i < rowPerThread; i = i + 1) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached0 * ACached.x + acc[i];
          acc[i] = BCached1 * ACached.y + acc[i];
          acc[i] = BCached2 * ACached.z + acc[i];
          ${t===3?"":"acc[i] = BCached3 * ACached.w + acc[i];"}
        }`,va=(e,t,r="f32",i,a=!1,n=32,s=!1,u=32)=>{let d=t[1]*e[1],p=t[0]*e[0],c=a?d:n,f=a?n:d,g=c/t[0],_=n/t[1];if(!((a&&g===4&&e[1]===4||!a&&(g===3||g===4))&&c%t[0]===0&&n%t[1]===0&&e[0]===4))throw new Error(`If transposeA ${a} is true, innerElementSize ${g} and workPerThread[1] ${e[1]} must be 4.
      Otherwise, innerElementSize ${g} must be 3 or 4.
  tileAWidth ${c} must be divisible by workgroupSize[0]${t[0]}. tileInner ${n} must be divisible by workgroupSize[1] ${t[1]}. colPerThread ${e[0]} must be 4.`);return`
var<workgroup> mm_Asub: array<array<vec${g}<${r}>, ${c/g}>, ${f}>;
var<workgroup> mm_Bsub: array<array<vec4<${r}>, ${p/e[0]}>, ${n}>;

const rowPerThread = ${e[1]};
const colPerThread = ${e[0]};
const innerElementSize = ${g};
const tileInner = ${n};

@compute @workgroup_size(${t[0]}, ${t[1]}, ${t[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
  let localRow = i32(localId.y);
  let tileRow = localRow * rowPerThread;
  let tileCol = i32(localId.x);

  let globalRow =i32(globalId.y) * rowPerThread;
  let globalCol = i32(globalId.x);
  let batch = ${s?"0":"i32(globalId.z)"};
  ${i?`let batchIndices = ${i.offsetToIndices("u32(batch)")};`:""}
  let globalRowStart = i32(workgroupId.y) * ${d};

  let num_tiles = ${s?`${Math.ceil(u/n)}`:"(uniforms.dim_inner - 1) / tileInner + 1"};
  var kStart = ${s?`i32(globalId.z) * ${u}`:"0"};

  var acc: array<vec4<${r}>, rowPerThread>;

  // Loop over shared dimension.
  let tileRowB = localRow * ${_};
  for (var t = 0; t < num_tiles; t = t + 1) {
      // Load one tile of A into local memory.
      for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
          let inputRow = tileRow + innerRow;
          let inputCol = tileCol;
          ${Su(a,i)}
      }

      // Load one tile of B into local memory.
      for (var innerRow = 0; innerRow < ${_}; innerRow = innerRow + 1) {
          let inputRow = tileRowB + innerRow;
          let inputCol = tileCol;
          mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol${i?", batchIndices":""});
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {
          let BCached0 = mm_Bsub[k * innerElementSize][tileCol];
          let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
          let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
          ${g===3?"":"let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];"}

          ${ku(a,g)}
      }

      workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
  }
}`},Pi=(e,t)=>e?`
            mm_Asub[inputRow][inputCol] = mm_readA(batch,
              kStart + inputRow,
              globalRowStart + inputCol${t?", batchIndices":""});
            `:`
            mm_Asub[inputRow][inputCol] = mm_readA(batch,
              globalRowStart + inputRow,
              kStart + inputCol${t?", batchIndices":""});
            `,Tu=e=>e?"let ACached = mm_Asub[k][tileRow + innerRow];":"let ACached = mm_Asub[tileRow + innerRow][k];",xa=(e,t,r="f32",i,a=!1,n=32,s=!1,u=32,d=!1)=>{let p=e[1]*t[1],c=e[0]*t[0],f=a?p:n,g=a?n:p;if(!(g%t[1]===0&&f%t[0]===0&&n%t[1]===0))throw new Error(`tileAHight ${g} must be divisible by workgroupSize[1]${t[1]}, tileAWidth ${f} must be divisible by workgroupSize[0]${t[0]}, tileInner ${n} must be divisible by workgroupSize[1]${t[1]}`);let _=g/t[1],w=f/t[0],b=n/t[1],S=d?`
    let localRow = i32(localId.y);
    let localCol = i32(localId.x);
    let globalRowStart = i32(workgroupId.y) * ${p};
    let globalColStart = i32(workgroupId.x) * ${c};

    // Loop over shared dimension.
    for (var t = 0; t < num_tiles; t = t + 1) {
      // Load one tile of A into local memory.
      for (var inputRow = localRow; inputRow < ${g}; inputRow = inputRow + ${t[1]}) {
        for (var inputCol = localCol; inputCol < ${f}; inputCol = inputCol + ${t[0]}) {
          ${Pi(a,i)}
        }
      }
      // Load one tile of B into local memory.
      for (var inputRow = localRow; inputRow < ${n}; inputRow = inputRow + ${t[1]}) {
            for (var inputCol = localCol; inputCol < ${c}; inputCol = inputCol + ${t[0]}) {
          mm_Bsub[inputRow][inputCol] = mm_readB(batch,
            kStart + inputRow,
            globalColStart + inputCol${i?", batchIndices":""});
        }
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      var BCached : array<${r}, colPerThread>;
      for (var k = 0; k < tileInner; k = k + 1) {
        for (var inner = 0; inner < colPerThread; inner = inner + 1) {
          BCached[inner] = mm_Bsub[k][localCol + inner * ${t[0]}];
        }
        for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
          let ACached = ${a?`mm_Asub[k][localRow + innerRow * ${t[1]}];`:`mm_Asub[localRow + innerRow * ${t[1]}][k];`}
          for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
            acc[innerRow][innerCol] = acc[innerRow][innerCol] +
                ACached * BCached[innerCol];
          }
        }
      }
      workgroupBarrier();
    }
    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      let gRow = globalRowStart + localRow + innerRow * ${t[1]};
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        let gCol = globalColStart + localCol + innerCol * ${t[0]};
        mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
      }
    }
    `:`
let tileRow = i32(localId.y) * rowPerThread;
let tileCol = i32(localId.x) * colPerThread;

let globalRow = i32(globalId.y) * rowPerThread;
let globalCol = i32(globalId.x) * colPerThread;
let globalRowStart = i32(workgroupId.y) * ${p};

let tileRowA = i32(localId.y) * ${_};
let tileColA = i32(localId.x) * ${w};
let tileRowB = i32(localId.y) * ${b};
// Loop over shared dimension.
for (var t = 0; t < num_tiles; t = t + 1) {
  // Load one tile of A into local memory.
  for (var innerRow = 0; innerRow < ${_}; innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < ${w}; innerCol = innerCol + 1) {
      let inputRow = tileRowA + innerRow;
      let inputCol = tileColA + innerCol;
      ${Pi(a,i)}
    }
  }

  // Load one tile of B into local memory.
  for (var innerRow = 0; innerRow < ${b}; innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
      let inputRow = tileRowB + innerRow;
      let inputCol = tileCol + innerCol;
      mm_Bsub[inputRow][inputCol] = mm_readB(batch,
        kStart + inputRow,
        globalCol + innerCol${i?", batchIndices":""});
    }
  }
  kStart = kStart + tileInner;
  workgroupBarrier();

  // Compute acc values for a single thread.
  var BCached : array<${r}, colPerThread>;
  for (var k = 0; k < tileInner; k = k + 1) {
    for (var inner = 0; inner < colPerThread; inner = inner + 1) {
      BCached[inner] = mm_Bsub[k][tileCol + inner];
    }

    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
      ${Tu(a)}
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
      }
    }
  }

  workgroupBarrier();
}

for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
  for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
    mm_write(batch, globalRow + innerRow, globalCol + innerCol,
        acc[innerRow][innerCol]);
  }
}
`;return`
  var<workgroup> mm_Asub : array<array<${r}, ${f}>, ${g}>;
  var<workgroup> mm_Bsub : array<array<${r}, ${c}>, ${n}>;
  const rowPerThread = ${e[1]};
  const colPerThread = ${e[0]};
  const tileInner = ${n};

@compute @workgroup_size(${t[0]}, ${t[1]}, ${t[2]})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = ${s?"0":"i32(globalId.z)"};
    ${i?`let batchIndices = ${i.offsetToIndices("u32(batch)")};`:""}
    let num_tiles = ${s?`${Math.ceil(u/n)}`:"(uniforms.dim_inner - 1) / tileInner + 1"};
    var kStart = ${s?`i32(globalId.z) * ${u}`:"0"};

    var acc : array<array<${r}, colPerThread>, rowPerThread>;
    ${S}
  }
`},Iu=(e,t,r,i,a=!1)=>{let[n,s,u,d]=i,p=ke(i[0].type.tensor);return`
    fn mm_readA(batch: i32, row: i32, colIn: i32, batchIndices: ${n.type.indices}) -> ${Ee(e,p)} {
      var value = ${Ee(e,p)}(0.0);
      let col = colIn * ${e};
      if(row < uniforms.dim_a_outer && col < uniforms.dim_inner)
      {
        var aIndices: ${s.type.indices};
        ${lr("aIndices",s,s.rank-2,n.rank,"batchIndices")}
        ${s.indicesSet("aIndices",s.rank-2,"u32(row)")}
        ${s.indicesSet("aIndices",s.rank-1,"u32(colIn)")}
        value = ${s.getByIndices("aIndices")};
      }
      return value;
    }

    fn mm_readB(batch: i32, row: i32, colIn: i32, batchIndices: ${n.type.indices}) -> ${Ee(e,p)} {
      var value = ${Ee(e,p)}(0.0);
      let col = colIn * ${e};
      if(row < uniforms.dim_inner && col < uniforms.dim_b_outer)
      {
        var bIndices: ${u.type.indices};
        ${lr("bIndices",u,u.rank-2,n.rank,"batchIndices")}
        ${u.indicesSet("bIndices",u.rank-2,"u32(row)")}
        ${u.indicesSet("bIndices",u.rank-1,"u32(colIn)")}
        value = ${u.getByIndices("bIndices")};
      }
      return value;
    }

    fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: ${Ee(e,p)}) {
      let col = colIn * ${e};
      if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer) {
        var value = valueIn;
        let coords = vec3<i32>(batch, row, colIn);
        ${t?`value = value + ${a?"bias[colIn]":`${Ee(e,p)}(bias[row])`};`:""}
        ${r}
        ${d.setByIndices("vec3<u32>(coords)","value")}
      }
    }
    `},Vr=(e,t,r,i,a=!1,n)=>{let s=e[0].dims,u=e[1].dims,d=s.slice(0,-2),p=u.slice(0,-2),c=i?i.slice(0,-2):r.slice(0,-2),f=B.size(c),g=s[s.length-2],_=s[s.length-1],w=u[u.length-1],b=_%4===0&&w%4===0,S=g<=8?[4,1,1]:[4,4,1],v=[8,8,1],$=[Math.ceil(w/v[0]/S[0]),Math.ceil(g/v[1]/S[1]),Math.ceil(f/v[2]/S[2])],I=b?4:1,T=[...d,g,_/I],E=T.length,A=[...p,_,w/I],C=A.length,O=[f,g,w/I],U=[{type:6,data:g},{type:6,data:w},{type:6,data:_}];Ot(t,U),U.push(...Q(c,T,A));let x=["rank","rank"],Y=e.length>2;Y&&(U.push(...Q(e[2].dims)),x.push("rank")),U.push(...Q(O));let G=V=>{let te=c.length,ee=Wa("batchDims",e[0].dataType,te,1),F=ke(e[0].dataType),R=M("a",e[0].dataType,E,I),q=M("b",e[1].dataType,C,I),X=j("result",e[0].dataType,O.length,I),_e=[R,q];if(Y){let ze=a?I:1;_e.push(M("bias",e[2].dataType,e[2].dims.length,ze))}let D=[{name:"dim_a_outer",type:"i32"},{name:"dim_b_outer",type:"i32"},{name:"dim_inner",type:"i32"}];Rt(t,D);let L=ke(X.type.tensor),K=At(t,X.type.value,L),re=Iu(I,Y,K,[ee,R,q,X],a);return`
  ${V.registerUniforms(D).registerInternalVariables(ee).declareVariables(..._e,X)}
  ${re}
  ${b?va(S,v,F,ee):xa(S,v,F,ee)}
                   `};return{name:"MatMul",shaderCache:{hint:`${S};${t.activation};${b};${a}`,inputDependencies:x},getRunData:()=>({outputs:[{dims:n?n(r):r,dataType:e[0].dataType}],dispatchGroup:{x:$[0],y:$[1],z:$[2]},programUniforms:U}),getShaderSource:G}}}),Eu,Wc,o0=P(()=>{J(),nt(),ae(),Nt(),Fa(),s0(),Za(),Eu=(e,t,r,i,a=!1,n,s=4,u=4,d=4,p="f32")=>{let c=U=>{switch(U){case 1:return"resData = x[xIndex];";case 3:return`resData = vec3<${p}>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);`;case 4:return"resData = x[xIndex / 4];";default:throw new Error(`innerElementSize ${U} is not supported.`)}},f=U=>{switch(U){case 1:return"return w[row * i32(uniforms.w_shape[3]) + colIn];";case 4:return"return w[row * i32(uniforms.w_shape[3]) / 4 + colIn];";default:throw new Error(`innerElementSize ${U} is not supported.`)}},g=e?`
    let coord = vec4<i32>(batch, xRow, xCol, xCh);
    `:`
    let coord = vec4<i32>(batch, xCh, xRow, xCol);
    `,_=e?`
    let coords = vec4<i32>(
      batch,
      row / outWidth,
      row % outWidth,
      col);
    `:`
    let coords = vec4<i32>(
      batch,
      row,
      col / outWidth,
      col % outWidth);
    `,w=e?"i32(uniforms.x_shape[1])":"i32(uniforms.x_shape[2])",b=e?"i32(uniforms.x_shape[2])":"i32(uniforms.x_shape[3])",S=e?"row":"col",v=e?"col":"row",$=`
    let inChannels = i32(uniforms.w_shape[2]);
    let outWidth = ${e?"i32(uniforms.result_shape[2])":"i32(uniforms.result_shape[3])"};
    let outRow = ${S} / outWidth;
    let outCol = ${S} % outWidth;

    let WRow = ${v} / (i32(uniforms.w_shape[1]) * inChannels);
    let WCol = ${v} / inChannels % i32(uniforms.w_shape[1]);
    let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
    let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];
    let xCh = ${v} % inChannels;
    var resData = ${Ee(s,p)}(0.0);
    // The bounds checking is always needed since we use it to pad zero for
    // the 'same' padding type.
    if (xRow >= 0 && xRow < ${w} && xCol >= 0 && xCol < ${b}) {
      ${g}
      let xIndex = getIndexFromCoords4D(coord, vec4<i32>(uniforms.x_shape));
      ${c(s)}
    }
    return resData;`,I=e?t&&i?`
    let col = colIn * ${s};
    ${$}`:`
    let col = colIn * ${s};
    if (row < uniforms.dim_a_outer && col < uniforms.dim_inner) {
      ${$}
    }
    return ${Ee(s,p)}(0.0);`:i&&r?`
    let col = colIn * ${s};
    ${$}`:`
    let col = colIn * ${s};
    if (row < uniforms.dim_inner && col < uniforms.dim_b_outer) {
      ${$}
    }
    return ${Ee(s,p)}(0.0);`,T=e?i&&r?f(u):`
    let col = colIn * ${u};
    if (row < uniforms.dim_inner && col < uniforms.dim_b_outer) {
      ${f(u)}
    }
    return ${Ee(u,p)}(0.0);`:`
    let col = colIn * ${u};
    if (row < uniforms.dim_inner && col < uniforms.dim_a_outer) {
      ${f(u)}
    }
    return ${Ee(u,p)}(0.0);`,E=Ee(d,p),A=Ee(e?s:u,p),C=Ee(e?u:s,p),O=At(n,E,p);return`
    fn mm_readA(batch: i32, row : i32, colIn : i32) -> ${A} {
      ${e?I:T}
    }

    fn mm_readB(batch: i32, row : i32, colIn : i32) -> ${C} {
      ${e?T:I}
    }

    fn mm_write(batch: i32, row : i32, colIn : i32, valueIn : ${E}) {
      let col = colIn * ${d};
      if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer)
      {
      var value = valueIn;
      let outWidth = ${e?"i32(uniforms.result_shape[2])":"i32(uniforms.result_shape[3])"};
      ${_}
      ${Pc(a)}
      ${O}
      setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
      }
    }`},Wc=(e,t,r,i,a,n,s,u,d)=>{let p=t.format==="NHWC",c=p?e[0].dims[3]:e[0].dims[1],f=r[0],g=p?r[2]:r[3],_=p?r[1]:r[2],w=p?r[3]:r[1],b=p&&(c%4===0||c%3===0)&&w%4===0,S=p?w:g*_,v=p?g*_:w,$=[8,8,1],I=i<=8?[4,1,1]:[4,4,1],T=[Math.ceil(S/$[0]/I[0]),Math.ceil(v/$[1]/I[1]),Math.ceil(f/$[2]/I[2])];le("verbose",()=>`[conv2d_mm_webgpu] dispatch = ${T}`);let E=b?p&&c%4!==0?3:4:1,A=$[1]*I[1],C=$[0]*I[0],O=Math.max($[0]*E,$[1]),U=i%A===0,x=a%C===0,Y=n%O===0,G=b?[E,4,4]:[1,1,1],V=[{type:6,data:i},{type:6,data:a},{type:6,data:n},{type:6,data:[t.pads[0],t.pads[1]]},{type:6,data:t.strides},{type:6,data:t.dilations}];Ot(t,V),V.push(...Q(e[0].dims,e[1].dims));let te=["rank","rank"];s&&(V.push(...Q(e[2].dims)),te.push("rank")),V.push(...Q(r));let ee=F=>{let R=[{name:"dim_a_outer",type:"i32"},{name:"dim_b_outer",type:"i32"},{name:"dim_inner",type:"i32"},{name:"pad",type:"i32",length:2},{name:"stride",type:"i32",length:2},{name:"dilation",type:"i32",length:2}];Rt(t,R);let q=b?4:1,X=ke(e[0].dataType),_e=`
      fn setOutputAtIndex(flatIndex : i32, value : ${b?`vec4<${X}>`:X}) {
        result[flatIndex] = ${b?`vec4<${X}>`:X}(value);
      }
      fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : ${b?`vec4<${X}>`:X}) {
        let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
        setOutputAtIndex(flatIndex ${b?"/ 4":""}, value);
      }`,D=M("x",e[0].dataType,e[0].dims.length,E===3?1:E),L=M("w",e[1].dataType,e[1].dims.length,q),K=[D,L],re=j("result",e[0].dataType,r.length,q);if(s){let ze=M("bias",e[2].dataType,e[2].dims.length,q);K.push(ze),_e+=`
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${b?`vec4<${X}>`:X} {
          return bias[coords.${p?"w":"y"}${b?"/ 4":""}];
        }`}return`
        ${qc("uniforms.result_strides")}
        //struct Uniforms { xShape : vec4<i32>, wShape : vec4<i32>, outShape : vec4<i32>,
        //  outShapeStrides: vec3<i32>, filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>,
        //  dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32 };
        ${F.registerUniforms(R).declareVariables(...K,re)}
        ${_e}
        ${Eu(p,U,x,Y,s,t,G[0],G[1],G[2],X)}
        ${b?va(I,$,X,void 0,!p,O):xa(I,$,X,void 0,!p,O,!1,void 0,u)}`};return{name:"Conv2DMatMul",shaderCache:{hint:`${t.cacheKey};${E};${b};${U};${x};${Y};${A};${C};${O}`,inputDependencies:te},getRunData:()=>({outputs:[{dims:d?d(r):r,dataType:e[0].dataType}],dispatchGroup:{x:T[0],y:T[1],z:T[2]},programUniforms:V}),getShaderSource:ee}}}),zu,qi,Jt,Cu,Wi,Au,Lc,Vc,u0=P(()=>{J(),nt(),ie(),ae(),Nt(),Fa(),zu=e=>{let t=1;for(let r=0;r<e.length;r++)t*=e[r];return t},qi=e=>typeof e=="number"?[e,e,e]:e,Jt=(e,t)=>t<=1?e:e+(e-1)*(t-1),Cu=(e,t,r,i=1)=>{let a=Jt(t,i);return Math.floor((e[0]*(r-1)-r+a)/2)},Wi=(e,t,r,i,a)=>{a==null&&(a=Cu(e,t[0],i[0]));let n=[0,0,0,r];for(let s=0;s<3;s++)e[s]+2*a>=t[s]&&(n[s]=Math.trunc((e[s]-t[s]+2*a)/i[s]+1));return n},Au=(e,t,r,i,a,n,s,u,d,p)=>{let c,f,g,_;if(e==="VALID"&&(e=0),typeof e=="number"){c={top:e,bottom:e,left:e,right:e,front:e,back:e};let w=Wi([t,r,i,1],[u,d,p],1,[a,n,s],e);f=w[0],g=w[1],_=w[2]}else if(Array.isArray(e)){if(!e.every((b,S,v)=>b===v[0]))throw Error(`Unsupported padding parameter: ${e}`);c={top:e[0],bottom:e[1],left:e[2],right:e[3],front:e[4],back:e[5]};let w=Wi([t,r,i,1],[u,d,p],1,[a,n,s],e[0]);f=w[0],g=w[1],_=w[2]}else if(e==="SAME_UPPER"){f=Math.ceil(t/a),g=Math.ceil(r/n),_=Math.ceil(i/s);let w=(f-1)*a+u-t,b=(g-1)*n+d-r,S=(_-1)*s+p-i,v=Math.floor(w/2),$=w-v,I=Math.floor(b/2),T=b-I,E=Math.floor(S/2),A=S-E;c={top:I,bottom:T,left:E,right:A,front:v,back:$}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:c,outDepth:f,outHeight:g,outWidth:_}},Lc=(e,t,r,i,a,n=!1,s="channelsLast")=>{let u,d,p,c,f;if(s==="channelsLast")[u,d,p,c,f]=e;else if(s==="channelsFirst")[u,f,d,p,c]=e;else throw new Error(`Unknown dataFormat ${s}`);let[g,,_,w,b]=t,[S,v,$]=qi(r),[I,T,E]=qi(i),A=Jt(_,I),C=Jt(w,T),O=Jt(b,E),{padInfo:U,outDepth:x,outHeight:Y,outWidth:G}=Au(a,d,p,c,S,v,$,A,C,O),V=n?g*f:g,te=[0,0,0,0,0];return s==="channelsFirst"?te=[u,V,x,Y,G]:s==="channelsLast"&&(te=[u,x,Y,G,V]),{batchSize:u,dataFormat:s,inDepth:d,inHeight:p,inWidth:c,inChannels:f,outDepth:x,outHeight:Y,outWidth:G,outChannels:V,padInfo:U,strideDepth:S,strideHeight:v,strideWidth:$,filterDepth:_,filterHeight:w,filterWidth:b,effectiveFilterDepth:A,effectiveFilterHeight:C,effectiveFilterWidth:O,dilationDepth:I,dilationHeight:T,dilationWidth:E,inShape:e,outShape:te,filterShape:t}},Vc=(e,t,r,i,a,n)=>{let s=n==="channelsLast";s?e[0].dims[3]:e[0].dims[1];let u=[64,1,1],d={x:r.map((S,v)=>v)},p=[Math.ceil(zu(d.x.map(S=>r[S]))/u[0]),1,1];le("verbose",()=>`[conv3d_naive_webgpu] dispatch = ${p}`);let c=1,f=B.size(r),g=[{type:12,data:f},{type:12,data:i},{type:12,data:a},{type:12,data:t.strides},{type:12,data:t.dilations}];Ot(t,g),g.push(...Q(e[0].dims,e[1].dims));let _=["rank","rank"],w=e.length===3;w&&(g.push(...Q(e[2].dims)),_.push("rank")),g.push(...Q(r));let b=S=>{let v=[{name:"output_size",type:"u32"},{name:"filter_dims",type:"u32",length:i.length},{name:"pads",type:"u32",length:a.length},{name:"strides",type:"u32",length:t.strides.length},{name:"dilations",type:"u32",length:t.dilations.length}];Rt(t,v);let $=1,I=ke(e[0].dataType),T=M("x",e[0].dataType,e[0].dims.length,c),E=M("W",e[1].dataType,e[1].dims.length,$),A=[T,E],C=j("result",e[0].dataType,r.length,$),O="";if(w){let Y=M("bias",e[2].dataType,e[2].dims.length,$);A.push(Y),O+=`
        fn getBiasByOutputCoords(coords : array<u32, 5>) -> ${I} {
          return bias[${s?Z("coords",4,5):Z("coords",1,5)}];
        }`}let U=Ee(c,I),x=At(t,U,I);return`
            ${O}
            fn getX(d0 : u32, d1 : u32, d2 : u32, d3 : u32, d4 : u32) -> f32 {
              let aIndices = array<u32, 5>(d0, d1, d2, d3, d4);
              return ${T.getByIndices("aIndices")};
            }
            fn getW(d0 : u32, d1 : u32, d2 : u32, d3 : u32, d4 : u32) -> f32 {
              let aIndices = array<u32, 5>(d0, d1, d2, d3, d4);
              return ${E.getByIndices("aIndices")};
            }
          ${S.registerUniforms(v).declareVariables(...A,C)}
          ${S.mainStart()}
          ${S.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
              let coords = ${C.offsetToIndices("global_idx")};
              let batch = ${Z("coords",0,T.rank)};
              let d2 = ${s?Z("coords",T.rank-1,T.rank):Z("coords",1,T.rank)};
              let xFRCCorner = vec3<u32>(${s?Z("coords",1,T.rank):Z("coords",2,T.rank)},
              ${s?Z("coords",2,T.rank):Z("coords",3,T.rank)},
              ${s?Z("coords",3,T.rank):Z("coords",4,T.rank)}) * uniforms.strides - uniforms.pads;
              let xFCorner = xFRCCorner.x;
              let xRCorner = xFRCCorner.y;
              let xCCorner = xFRCCorner.z;
              let xShapeY = ${s?Z("uniforms.x_shape",1,T.rank):Z("uniforms.x_shape",2,T.rank)};
              let xShapeZ = ${s?Z("uniforms.x_shape",2,T.rank):Z("uniforms.x_shape",3,T.rank)};
              let xShapeW = ${s?Z("uniforms.x_shape",3,T.rank):Z("uniforms.x_shape",4,T.rank)};
              let xShapeU = ${s?Z("uniforms.x_shape",4,T.rank):Z("uniforms.x_shape",1,T.rank)};
              let inputDepthNearestVec4 = (xShapeU / 4) * 4;
              let inputDepthVec4Remainder = xShapeU % 4;

              var value = 0.0;
              for (var wF = 0u; wF < uniforms.filter_dims[0]; wF++) {
                let xF = xFCorner + wF * uniforms.dilations[0];
                if (xF < 0 || xF >= xShapeY) {
                  continue;
                }

                for (var wR = 0u; wR < uniforms.filter_dims[1]; wR++) {
                  let xR = xRCorner + wR * uniforms.dilations[1];
                  if (xR < 0 || xR >= xShapeZ) {
                    continue;
                  }

                  for (var wC = 0u; wC < uniforms.filter_dims[2]; wC++) {
                    let xC = xCCorner + wC * uniforms.dilations[2];
                    if (xC < 0 || xC >= xShapeW) {
                      continue;
                    }

                    for (var d1 = 0u; d1 < inputDepthNearestVec4; d1 += 4) {
                      ${s?`let xValues = vec4<f32>(
                               getX(batch, xF, xR, xC, d1),
                               getX(batch, xF, xR, xC, d1 + 1),
                               getX(batch, xF, xR, xC, d1 + 2),
                               getX(batch, xF, xR, xC, d1 + 3));
                            `:`let xValues = vec4<f32>(
                               getX(batch, d1, xF, xR, xC),
                               getX(batch, d1 + 1, xF, xR, xC),
                               getX(batch, d1 + 2, xF, xR, xC),
                               getX(batch, d1 + 3, xF, xR, xC));
                            `}
                            let wValues = vec4<f32>(
                              getW(d2, d1, wF, wR, wC),
                              getW(d2, d1 + 1, wF, wR, wC),
                              getW(d2, d1 + 2, wF, wR, wC),
                              getW(d2, d1 + 3, wF, wR, wC));
                      value += dot(xValues, wValues);
                    }
                    if (inputDepthVec4Remainder == 1) {
                        ${s?`value += getX(batch, xF, xR, xC, inputDepthNearestVec4)
                          * getW(d2, inputDepthNearestVec4, wF, wR, wC);`:`value += getX(batch, inputDepthNearestVec4, xF, xR, xC)
                          * getW(d2, inputDepthNearestVec4, wF, wR, wC);`}
                    } else if (inputDepthVec4Remainder == 2) {
                      ${s?`let xValues = vec2<f32>(
                        getX(batch, xF, xR, xC, inputDepthNearestVec4),
                        getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1));
                      `:`let xValues = vec2<f32>(
                        getX(batch, inputDepthNearestVec4, xF, xR, xC),
                        getX(batch, inputDepthNearestVec4 + 1, xF, xR, xC));
                    `}
                    let wValues = vec2<f32>(
                      getW(d2, inputDepthNearestVec4, wF, wR, wC),
                      getW(d2, inputDepthNearestVec4 + 1, wF, wR, wC));
                      value += dot(xValues, wValues);
                    } else if (inputDepthVec4Remainder == 3) {
                      ${s?`let xValues = vec3<f32>(
                        getX(batch, xF, xR, xC, inputDepthNearestVec4),
                        getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1),
                        getX(batch, xF, xR, xC, inputDepthNearestVec4 + 2));
                      `:`let xValues = vec3<f32>(
                        getX(batch, inputDepthNearestVec4, xF, xR, xC),
                        getX(batch, inputDepthNearestVec4 + 1, xF, xR, xC),
                        getX(batch, inputDepthNearestVec4 + 2, xF, xR, xC));
                    `}
                    let wValues = vec3<f32>(
                      getW(d2, inputDepthNearestVec4, wF, wR, wC),
                      getW(d2, inputDepthNearestVec4 + 1, wF, wR, wC),
                      getW(d2, inputDepthNearestVec4 + 2, wF, wR, wC));
                      value += dot(xValues, wValues);
                    }
                  }
                }
              }
              ${w?"value = value + getBiasByOutputCoords(coords)":""};
              ${x}
              result[global_idx] = f32(value);
          }`};return{name:"Conv3DNaive",shaderCache:{hint:`${t.cacheKey};${s};${c};${w}`,inputDependencies:_},getRunData:()=>({outputs:[{dims:r,dataType:e[0].dataType}],dispatchGroup:{x:p[0],y:p[1],z:p[2]},programUniforms:g}),getShaderSource:b}}}),Gc,Hc,l0=P(()=>{J(),ie(),ae(),Nt(),Gc=(e,t,r,i)=>{let a=e.length>2,n=a?"value += b[output_channel];":"",s=e[0].dims,u=e[1].dims,d=t.format==="NHWC",p=d?r[3]:r[1],c=p/t.group,f=d&&c>=4?$e(p):1,g=B.size(r)/f,_=[{type:12,data:g},{type:12,data:t.dilations},{type:12,data:[t.strides[0],t.strides[1]]},{type:12,data:[t.pads[0],t.pads[1]]},{type:12,data:c}];Ot(t,_),_.push(...Q(s,[u[0],u[1],u[2],u[3]/f]));let w=a?["rank","rank","rank"]:["rank","rank"];_.push(...Q([r[0],r[1],r[2],r[3]/f]));let b=S=>{let v=j("output",e[0].dataType,r.length,f),$=ke(v.type.tensor),I=At(t,v.type.value,$),T=M("x",e[0].dataType,s.length),E=M("w",e[1].dataType,u.length,f),A=[T,E];a&&A.push(M("b",e[2].dataType,e[2].dims,f));let C=[{name:"output_size",type:"u32"},{name:"dilations",type:"u32",length:t.dilations.length},{name:"strides",type:"u32",length:2},{name:"pads",type:"u32",length:2},{name:"output_channels_per_group",type:"u32"}];Rt(t,C);let O=d?`
      for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[0]; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];

        if (xHeight < 0u || xHeight >= uniforms.x_shape[1]) {
          continue;
        }

        for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[1]; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];
          if (xWidth < 0u || xWidth >= uniforms.x_shape[2]) {
            continue;
          }

          for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[2]; wInChannel++) {
            let input_channel = in_channel_offset + wInChannel;
            let xVal = ${T.get("batch","xHeight","xWidth","input_channel")};
            let wVal = ${E.get("wHeight","wWidth","wInChannel","output_channel")};
            value += xVal * wVal;
          }
        }
      }
      `:`
      for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[1]; wInChannel++) {
        let input_channel = in_channel_offset + wInChannel;
        for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[2]; wHeight++) {
          let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];

          if (xHeight < 0u || xHeight >= uniforms.x_shape[2]) {
            continue;
          }

          for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[3]; wWidth++) {
            let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];
            if (xWidth < 0u || xWidth >= uniforms.x_shape[3]) {
              continue;
            }

            let xVal = ${T.get("batch","input_channel","xHeight","xWidth")};
            let wVal = ${E.get("output_channel","wInChannel","wHeight","wWidth")};
            value += xVal * wVal;
          }
        }
      }
      `;return`
  ${S.registerUniforms(C).declareVariables(...A,v)}

  ${S.mainStart()}
    ${S.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}

    let outputIndices = ${v.offsetToIndices("global_idx")};
    let batch: u32 = outputIndices[0];
    let output_channel: u32 = outputIndices[${d?3:1}];
    let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[${d?1:2}], outputIndices[${d?2:3}]) * uniforms.strides - uniforms.pads;
    let group_id: u32 = output_channel * ${f} / uniforms.output_channels_per_group;
    var in_channel_offset = group_id * uniforms.w_shape[${d?2:1}];

    var value: ${v.type.value} = ${v.type.value}(0);
    ${O}
    ${n}
    ${I}
    ${v.setByOffset("global_idx","value")}
  }`};return{name:"GroupedConv",shaderCache:{hint:`${t.cacheKey}_${f}`,inputDependencies:w},getRunData:()=>({outputs:[{dims:i?i(r):r,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(g/64)},programUniforms:_}),getShaderSource:b}},Hc=(e,t,r,i)=>{let a=e.length>2,n=$e(r[3]),s=$e(r[2]),u=B.size(r)/n/s,d=[e[0].dims[0],e[0].dims[1],e[0].dims[2],e[0].dims[3]/n],p=[e[1].dims[0],e[1].dims[1],e[1].dims[2],e[1].dims[3]/n],c=[r[0],r[1],r[2],r[3]/n],f=[{type:12,data:u},{type:6,data:[t.strides[0],t.strides[1]]},{type:6,data:[t.pads[0],t.pads[1]]}];Ot(t,f),f.push(...Q(d,p,c));let g=(s-1)*t.strides[1]+p[1],_=w=>{let b=j("output",e[0].dataType,c.length,n),S=ke(b.type.tensor),v=At(t,b.type.value,S),$=M("x",e[0].dataType,d.length,n),I=M("w",e[1].dataType,p.length,n),T=[$,I];a&&T.push(M("b",e[2].dataType,e[2].dims,n));let E=a?"value += b[output_channel];":"",A=[{name:"output_size",type:"u32"},{name:"strides",type:"i32",length:2},{name:"pads",type:"i32",length:2}];return Rt(t,A),`
  ${w.registerUniforms(A).declareVariables(...T,b)}
  ${w.mainStart()}
    ${w.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    let width0 = uniforms.output_shape[3];
    let output_channel = global_idx % width0;
    var index1 = global_idx / width0;
    let width1 = uniforms.output_shape[2] / ${s}u;
    let col = (index1 % width1) * ${s}u;
    index1 = index1 / width1;
    let row = index1 % uniforms.output_shape[1];
    let batch = index1 / uniforms.output_shape[1];

    let x_corner = vec2<i32>(i32(row), i32(col)) * uniforms.strides - uniforms.pads;

    var x_vals: array<${$.type.value}, ${g}>;
    var values: array<${b.type.value}, ${s}>;
    let input_channel = output_channel;
    // Use constant instead of uniform can give better performance for w's height/width.
    for (var w_height: u32 = 0u; w_height < ${p[0]}; w_height++) {
      let x_height = x_corner.x + i32(w_height);
      if (x_height >= 0 && u32(x_height) < uniforms.x_shape[1]) {
        for (var i = 0; i < ${g}; i++) {
          let x_width = x_corner.y + i;
          if (x_width >= 0 && u32(x_width) < uniforms.x_shape[2]) {
            x_vals[i] = ${$.get("batch","u32(x_height)","u32(x_width)","input_channel")};
          } else {
            x_vals[i] = ${$.type.value}(0);
          }
        }
        for (var w_width: u32 = 0u; w_width < ${p[1]}; w_width++) {
          let w_val = ${I.get("w_height","w_width","0","output_channel")};
          for (var i = 0u; i < ${s}u; i++) {
            values[i] = fma(x_vals[i * u32(uniforms.strides[1]) + w_width], w_val, values[i]);
          }
        }
      }
    }

    for (var i = 0u; i < ${s}u; i++) {
      var value = values[i];
      ${E}
      ${v}
      ${b.set("batch","row","col + i","output_channel","value")};
    }
  }`};return{name:"GroupedConv-Vectorize",shaderCache:{hint:`${t.cacheKey};${n};${s};${g};${p[0]};${p[1]}`,inputDependencies:a?["rank","rank","type"]:["rank","rank"]},getRunData:()=>({outputs:[{dims:i?i(r):r,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(u/64)},programUniforms:f}),getShaderSource:_}}}),Ou,Cr,Ru,Ar,Sa,Li,Bu,Nu,ka,d0=P(()=>{ie(),o0(),u0(),Za(),l0(),Nt(),Ka(),yt(),Ou=(e,t,r,i,a,n)=>{let s=e[0],u=e.slice(n?1:2,n?3:4),d=u.length,p=t[0],c=t.slice(2).map((g,_)=>g+(g-1)*(r[_]-1)),f=u.map((g,_)=>g+i[_]+i[_+d]).map((g,_)=>Math.floor((g-c[_]+a[_])/a[_]));return f.splice(0,0,s),f.splice(n?3:1,0,p),f},Cr=[2,3,1,0],Ru=(e,t)=>{if(!e||e.length!==2&&e.length!==3)throw new Error("Conv requires 2 or 3 inputs");if(e[0].dims.length>5)throw new Error("greater than 5D is not supported");if(e[0].dims.length!==e[1].dims.length)throw new Error("filter does not have same dimension as input");let r=e[0].dims[t.format==="NHWC"?e[0].dims.length-1:1],i=e[1].dims[1]*t.group;if(r!==i)throw new Error("FILTER_IN_CHANNEL should be equal to DATA_CHANNEL");if(e.length===3&&(e[2].dims.length!==1||e[1].dims[0]!==e[2].dims[0]))throw new Error("invalid bias");let a=e[0].dims.length-2;if(t.dilations.length!==a)throw new Error(`dilations should be ${a}D`);if(t.strides.length!==a)throw new Error(`strides should be ${a}D`);if(t.pads.length!==a*2)throw new Error(`pads should be ${a*2}D`);if(t.kernelShape.length!==0&&t.kernelShape.length!==e[1].dims.length-2)throw new Error("invalid kernel shape")},Ar=(e,t)=>{let r=e.kernelShape.slice();r.length<t[1].dims.length-2&&r.push(...Array(t[1].dims.length-2-r.length).fill(0));for(let n=2;n<t[1].dims.length;++n)r[n-2]===0&&(r[n-2]=t[1].dims[n]);let i=e.pads.slice();Wr.adjustPadsBasedOnAutoPad(t[0].dims,e.strides,e.dilations,r,i,e.format==="NHWC",e.autoPad);let a=Object.assign({},e);return Object.assign(a,{kernelShape:r,pads:i}),a},Sa=e=>{let t=Ha(e),r=e.format,i=["NOTSET","VALID","SAME_UPPER","SAME_LOWER"][e.auto_pad],a=e.dilations,n=e.group,s=e.kernel_shape,u=e.pads,d=e.strides,p=e.w_is_const();return{autoPad:i,format:r,dilations:a,group:n,kernelShape:s,pads:u,strides:d,wIsConst:p,...t,cacheKey:`${e.format};${t.activation};`}},Li=(e,t,r,i)=>{let a=r.format==="NHWC",n=Ou(t[0].dims,t[1].dims,r.dilations,r.pads,r.strides,a);if(r.group!==1){let A=[t[0]];if(a){let C=e.kernelCustomData.wT??e.compute(De(t[1],Cr),{inputs:[1],outputs:[r.wIsConst?-2:-1]})[0];r.wIsConst&&!e.kernelCustomData.wT&&(e.kernelCustomData.wT=C),A.push(C)}else A.push(t[1]);t.length===3&&A.push(t[2]),!e.adapterInfo.isArchitecture("ampere")&&a&&t[1].dims[0]===r.group&&t[1].dims[1]===1&&r.dilations[0]===1&&r.dilations[1]===1?e.compute(Hc(A,r,n,i),{inputs:A}):e.compute(Gc(A,r,n,i),{inputs:A});return}let s=t.length===3,u=t[0].dims[a?1:2],d=t[0].dims[a?2:3],p=t[0].dims[a?3:1],c=t[1].dims[2],f=t[1].dims[3],g=n[a?1:2],_=n[a?2:3],w=n[a?3:1],b=a&&c===u&&f===d&&r.pads[0]===0&&r.pads[1]===0;if(b||c===1&&f===1&&r.dilations[0]===1&&r.dilations[1]===1&&r.strides[0]===1&&r.strides[1]===1&&r.pads[0]===0&&r.pads[1]===0){let A=n[0],C,O,U,x=[];if(a){let V=e.kernelCustomData.wT??e.compute(De(t[1],Cr),{inputs:[1],outputs:[r.wIsConst?-2:-1]})[0];if(r.wIsConst&&!e.kernelCustomData.wT&&(e.kernelCustomData.wT=V),b){let te=u*d*p;C=t[0].reshape([1,A,te]),O=V.reshape([1,te,w]),U=[1,A,w]}else C=t[0].reshape([A,u*d,p]),O=V.reshape([1,p,w]),U=[A,g*_,w];x.push(C),x.push(O)}else C=t[0].reshape([A,p,u*d]),O=t[1].reshape([1,w,p]),U=[A,w,g*_],x.push(O),x.push(C);s&&x.push(t[2]);let Y=U[2],G=x[0].dims[x[0].dims.length-1];Y<8&&G<8?e.compute(ja(x,r,n,U,a,i),{inputs:x}):e.compute(Vr(x,r,n,U,a,i),{inputs:x});return}let S=!0,v=e.kernelCustomData.wT??e.compute(De(t[1],Cr),{inputs:[1],outputs:[r.wIsConst?-2:-1]})[0];r.wIsConst&&!e.kernelCustomData.wT&&(e.kernelCustomData.wT=v);let $=[t[0],v];s&&$.push(t[2]);let I=a?g*_:w,T=a?w:g*_,E=c*f*p;e.compute(Wc($,r,n,I,T,E,s,S,i),{inputs:$})},Bu=(e,t)=>{let r=t.format==="NHWC",i=[e.inputs[0].reshape(r?[e.inputs[0].dims[0],1,e.inputs[0].dims[1],e.inputs[0].dims[2]]:[e.inputs[0].dims[0],e.inputs[0].dims[1],1,e.inputs[0].dims[2]]),e.inputs[1].reshape([e.inputs[1].dims[0],e.inputs[1].dims[1],1,e.inputs[1].dims[2]])];e.inputs.length===3&&i.push(e.inputs[2]);let a=[0,t.pads[0],0,t.pads[1]],n=[1].concat(t.strides),s=[1].concat(t.dilations),u=[1].concat(t.kernelShape),d=Ar({...t,pads:a,strides:n,dilations:s,kernelShape:u},i);Li(e,i,d,p=>r?[p[0],p[2],p[3]]:[p[0],p[1],p[3]])},Nu=(e,t,r)=>{let i=r.format==="NHWC"?"channelsLast":"channelsFirst",a=Ar(r,t),n=r.autoPad==="NOTSET"?r.pads:r.autoPad,s=Lc(t[0].dims,t[1].dims,r.strides,r.dilations,n,!1,i);e.compute(Vc(t,a,s.outShape,[s.filterDepth,s.filterHeight,s.filterWidth],[s.padInfo.front,s.padInfo.top,s.padInfo.left],i))},ka=(e,t)=>{if(Ru(e.inputs,t),e.inputs[0].dims.length===3)Bu(e,t);else if(e.inputs[0].dims.length===5)Nu(e,e.inputs,t);else{let r=Ar(t,e.inputs);Li(e,e.inputs,r)}}}),Fc,p0=P(()=>{J(),nt(),ie(),ae(),Fc=(e,t,r)=>{let i=e.length>2,a=t.outputShape,n=t.format==="NHWC",s=t.group,u=e[1].dims,d=u[2]/s,p=u[3],c=n?$e(d):1,f=n&&p===1&&d>=4,g=f?Math.floor(d/4)*4:Math.floor(d/c)*c,_=d-g,w=n?$e(p):1,b=n?p===1?c:w:1,S=B.size(a)/w,v=[Math.ceil(S/64),1,1];le("verbose",()=>`[conv2d_backprop_webgpu] dispatch = ${v}`);let $=["rank","rank"],I=[t.strides[0],t.strides[1]],T=[t.kernelShape[n?1:2],t.kernelShape[n?2:3]],E=[t.dilations[0],t.dilations[1]],A=[T[0]+(t.dilations[0]<=1?0:(t.kernelShape[n?1:2]-1)*(t.dilations[0]-1)),T[1]+(t.dilations[1]<=1?0:(t.kernelShape[n?2:3]-1)*(t.dilations[1]-1))],C=[A[0]-1-Math.floor((t.pads[0]+t.pads[2])/2),A[1]-1-Math.floor((t.pads[1]+t.pads[3])/2)],O=[{type:12,data:S},{type:12,data:I},{type:12,data:T},{type:12,data:E},{type:12,data:A},{type:6,data:C},{type:12,data:g},{type:12,data:d},{type:12,data:p},...Q(e[0].dims,e[1].dims)];i&&(O.push(...Q(e[2].dims)),$.push("rank")),O.push(...Q(a));let U=x=>{let Y=[{name:"output_size",type:"u32"},{name:"strides",type:"u32",length:I.length},{name:"filter_dims",type:"u32",length:T.length},{name:"dilations",type:"u32",length:T.length},{name:"effective_filter_dims",type:"u32",length:A.length},{name:"pads",type:"i32",length:C.length},{name:"input_channels_per_group_int",type:"u32"},{name:"input_channels_per_group",type:"u32"},{name:"output_channels_per_group",type:"u32"}],G=ke(e[0].dataType),V=n?1:2,te=n?2:3,ee=n?3:1,F=M("W",e[1].dataType,e[1].dims.length,b),R=M("Dy",e[0].dataType,e[0].dims.length,c),q=[R,F];i&&q.push(M("bias",e[2].dataType,[a[ee]].length,w));let X=j("result",e[0].dataType,a.length,w),_e=()=>{let K="";if(f)c===4?K+=`
        let xValue = ${R.getByOffset("x_offset")};
        let wValue = ${F.getByOffset("w_offset")};
        dotProd = dotProd + dot(xValue, wValue);
        x_offset += 1u;
        w_offset += 1u;`:c===2?K+=`
          dotProd = dotProd + dot(vec4<${G}>(${R.getByOffset("x_offset")}, ${R.getByOffset("x_offset + 1u")}), vec4<${G}>(${F.getByOffset("w_offset")}, ${F.getByOffset("w_offset + 1u")}));
          x_offset += 2u;
          w_offset += 2u;`:c===1&&(K+=`
          dotProd = dotProd + dot(vec4<${G}>(${R.getByOffset("x_offset")}, ${R.getByOffset("x_offset + 1u")}, ${R.getByOffset("x_offset + 2u")}, ${R.getByOffset("x_offset + 3u")}), vec4<${G}>(${F.getByOffset("w_offset")}, ${F.getByOffset("w_offset + 1u")}, ${F.getByOffset("w_offset + 2u")}, ${F.getByOffset("w_offset + 3u")}));
          x_offset += 4u;
          w_offset += 4u;`);else if(K+=`
                  let xValue = ${n?R.getByOffset(`${R.indicesToOffset(`${R.type.indices}(batch, idyR, idyC, inputChannel)`)} / ${c}`):R.get("batch","inputChannel","idyR","idyC")};
        `,c===1)K+=`
          let w_offset = ${F.indicesToOffset(`${F.type.indices}(u32(wRPerm), u32(wCPerm), inputChannel, wOutChannel)`)};
          let wValue = ${F.getByOffset(`w_offset / ${b}`)};
          dotProd = dotProd + xValue * wValue;`;else for(let re=0;re<c;re++)K+=`
            let wValue${re} = ${F.getByOffset(`${F.indicesToOffset(`${F.type.indices}(u32(wRPerm), u32(wCPerm), inputChannel + ${re}, wOutChannel)`)} / ${b}`)};
            dotProd = dotProd + xValue[${re}] * wValue${re};`;return K},D=()=>{if(_===0)return"";if(!f)throw new Error(`packInputAs4 ${f} is not true.`);let K="";if(c===1){K+="dotProd = dotProd";for(let re=0;re<_;re++)K+=`
            + ${R.getByOffset(`x_offset + ${re}`)} * ${F.getByOffset(`w_offset + ${re}`)}`;K+=";"}else if(c===2){if(_!==2)throw new Error(`Invalid inputChannelsRemainder ${_}.`);K+=`
          let xValue = ${R.getByOffset("x_offset")};
          let wValue = ${F.getByOffset("w_offset")};
          dotProd = dotProd + dot(xValue, wValue);`}return K},L=`
            let outputIndices = ${X.offsetToIndices(`global_idx * ${w}`)};
            let batch = ${X.indicesGet("outputIndices",0)};
            let d1 = ${X.indicesGet("outputIndices",ee)};
            let r = ${X.indicesGet("outputIndices",V)};
            let c = ${X.indicesGet("outputIndices",te)};
            let dyCorner = vec2<i32>(i32(r), i32(c)) - uniforms.pads;
            let dyRCorner = dyCorner.x;
            let dyCCorner = dyCorner.y;
            let groupId = d1 / uniforms.output_channels_per_group;
            let wOutChannel = d1 - groupId * uniforms.output_channels_per_group;
            // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
            // ? = to be determined. : = across all values in that axis.
            var dotProd = ${X.type.value}(0.0);
            var wR: u32 = 0;
            if (uniforms.dilations.x == 1) {
              // Minimum wR >= 0 that satisfies (dyRCorner + wR) % (uniforms.strides.x) == 0
              wR = u32(((dyRCorner + i32(uniforms.strides.x) - 1) / i32(uniforms.strides.x)) * i32(uniforms.strides.x) - dyRCorner);
            }
            for (; wR < uniforms.effective_filter_dims.x; wR = wR + 1) {
              if (wR % uniforms.dilations.x != 0) {
                continue;
              }
              let dyR = (${G}(dyRCorner) + ${G}(wR)) / ${G}(uniforms.strides[0]);
              let wRPerm = uniforms.filter_dims.x - 1 - wR / uniforms.dilations.x;
              if (dyR < 0.0 || dyR >= ${G}(uniforms.Dy_shape[${V}]) || fract(dyR) > 0.0 ||
                  wRPerm < 0) {
                continue;
              }
              let idyR: u32 = u32(dyR);
              var wC: u32 = 0;
              if (uniforms.dilations.y == 1) {
                // Minimum wC >= 0 that satisfies (dyCCorner + wC) % (uniforms.strides.y) == 0
                wC = u32(((dyCCorner + i32(uniforms.strides.y) - 1) / i32(uniforms.strides.y)) * i32(uniforms.strides.y) - dyCCorner);
              }
              for (; wC < uniforms.effective_filter_dims.y; wC = wC + 1) {
                if (wC % uniforms.dilations.y != 0) {
                  continue;
                }
                let dyC = (${G}(dyCCorner) + ${G}(wC)) / ${G}(uniforms.strides.y);
                let wCPerm = uniforms.filter_dims.y - 1 - wC / uniforms.dilations.y;
                if (dyC < 0.0 || dyC >= ${G}(uniforms.Dy_shape[${te}]) ||
                    fract(dyC) > 0.0 || wCPerm < 0) {
                  continue;
                }
                let idyC: u32 = u32(dyC);
                var inputChannel = groupId * uniforms.input_channels_per_group;
                ${f?`
                var x_offset = ${R.indicesToOffset(`${R.type.indices}(batch, idyR, idyC, inputChannel)`)} / ${c};
                var w_offset = ${F.indicesToOffset(`${F.type.indices}(wRPerm, wCPerm, inputChannel, wOutChannel)`)} / ${b};
                  `:""}
                for (var d2: u32 = 0; d2 < uniforms.input_channels_per_group_int; d2 = d2 + ${f?4:c}) {
                  ${_e()}
                  inputChannel = inputChannel + ${f?4:c};
                }
                ${D()}
                wC = wC + uniforms.strides.y - 1;
              }
              wR = wR + uniforms.strides[0] - 1;
            }
            let value = dotProd${i?` + bias[d1 / ${w}]`:""};
            ${X.setByOffset("global_idx","value")};
          `;return`
    ${x.registerUniforms(Y).declareVariables(...q,X)}
      ${x.mainStart()}
      ${x.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")};
    ${L}}`};return{name:"ConvTranspose2D",shaderCache:{hint:`${t.cacheKey};${c}${b}${w}${f}${_}`,inputDependencies:$},getRunData:()=>({dispatchGroup:{x:v[0],y:v[1],z:v[2]},outputs:[{dims:r?r(a):a,dataType:e[0].dataType}],programUniforms:O}),getShaderSource:U}}}),Du,Mu,Uu,Vi,jc,Pu,Gi,qu,Kc,c0=P(()=>{p0(),Nt(),yt(),Du=(e,t,r,i,a,n)=>(e-1)*t+r+(i-1)*a+1-n,Mu=(e,t,r,i,a)=>{let n=Math.floor(e/2);t==="SAME_UPPER"?(r[i]=n,r[a]=e-n):t==="SAME_LOWER"&&(r[i]=e-n,r[a]=n)},Uu=(e,t,r,i,a,n,s,u,d,p)=>{let c=e.length-2,f=p.length===0;d.length<c&&d.push(...Array(c-d.length).fill(0));let g=e[0],_=t[u?3:1]*a;for(let w=0,b=e.length-c-(u?1:0);w<c;++w,++b){let S=e[b],v=f?S*s[w]:p[w],$=Du(S,s[w],n[w],t[b],r[w],v);Mu($,i,n,w,w+c),f&&p.push(s[w]*(S-1)+d[w]+(t[b]-1)*r[w]+1-n[w]-n[w+c])}p.splice(0,0,g),p.splice(u?3:1,0,_)},Vi=(e,t)=>{let r=e.kernelShape.slice();if(e.kernelShape.length===0||e.kernelShape.reduce((f,g)=>f*g,1)===0){r.length=0;for(let f=2;f<t[1].dims.length;++f)r.push(t[1].dims[f])}let i=e.format==="NHWC";r.splice(0,0,t[1].dims[0]),r.splice(i?3:1,0,t[1].dims[1]);let a=e.pads.slice(),n=e.outputShape.slice(),s=e.outputPadding.slice(),u=t[0].dims,d=e.dilations.slice();if(d.reduce((f,g)=>f+g,0)===0){let f=t[0].dims.length-2;d=new Array(f).fill(1)}let p=e.strides.slice();if(p.reduce((f,g)=>f+g,0)===0){let f=t[0].dims.length-2;p=new Array(f).fill(1)}Uu(u,r,d,e.autoPad,e.group,a,p,i,s,n);let c=Object.assign({},e);return Object.assign(c,{kernelShape:r,pads:a,outputPadding:s,outputShape:n,dilations:d,strides:p}),c},jc=e=>{let t=Ha(e),r=e.format,i=["NOTSET","VALID","SAME_UPPER","SAME_LOWER"][typeof e.autoPad>"u"?0:e.autoPad],a=e.dilations,n=e.group,s=e.kernelShape,u=e.pads,d=e.strides,p=e.wIsConst(),c=e.outputPadding,f=e.outputShape;return{autoPad:i,format:r,dilations:a,group:n,kernelShape:s,outputPadding:c,outputShape:f,pads:u,strides:d,wIsConst:p,...t,cacheKey:`${e.format};${t.activation};`}},Pu=(e,t)=>{if(!e||e.length!==2&&e.length!==3)throw new Error("Conv requires 2 or 3 inputs");if(e[0].dims.length!==4&&e[0].dims.length!==3)throw new Error("currently only support 2-dimensional conv");if(e[0].dims.length!==e[1].dims.length)throw new Error("filter does not have same dimension as input");let r=e[0].dims[t.format==="NHWC"?e[0].dims.length-1:1],i=e[1].dims[0];if(r!==i)throw new Error("FILTER_IN_CHANNEL should be equal to DATA_CHANNEL");let a=e[1].dims[1]*t.group;if(e.length===3&&(e[2].dims.length!==1||e[2].dims[0]!==a))throw new Error("invalid bias");let n=e[0].dims.length-2;if(t.dilations.reduce((s,u)=>s+u,0)>0&&t.dilations.length!==n)throw new Error(`dilations should be ${n}D`);if(t.strides.reduce((s,u)=>s+u,0)>0&&t.strides.length!==n)throw new Error(`strides should be ${n}D`);if(t.pads.reduce((s,u)=>s+u,0)>0&&t.pads.length!==n*2)throw new Error(`pads should be ${n*2}D`);if(t.outputPadding.length!==n&&t.outputPadding.length!==0)throw new Error(`output_padding should be ${n}D`);if(t.kernelShape.reduce((s,u)=>s+u,0)>0&&t.kernelShape.length!==0&&t.kernelShape.length!==e[1].dims.length-2)throw new Error("invalid kernel shape");if(t.outputShape.length!==0&&t.outputShape.length!==e[0].dims.length-2)throw new Error("invalid output shape")},Gi=(e,t,r,i)=>{let a=e.kernelCustomData.wT??e.compute(De(t[1],[2,3,0,1]),{inputs:[1],outputs:[r.wIsConst?-2:-1]})[0];r.wIsConst&&!e.kernelCustomData.wT&&(e.kernelCustomData.wT=a);let n=[t[0],a];t.length===3&&n.push(t[2]),e.compute(Fc(n,r,i),{inputs:n})},qu=(e,t)=>{let r=t.format==="NHWC",i=[e.inputs[0].reshape(r?[e.inputs[0].dims[0],1,e.inputs[0].dims[1],e.inputs[0].dims[2]]:[e.inputs[0].dims[0],e.inputs[0].dims[1],1,e.inputs[0].dims[2]]),e.inputs[1].reshape([e.inputs[1].dims[0],e.inputs[1].dims[1],1,e.inputs[1].dims[2]])];e.inputs.length===3&&i.push(e.inputs[2]);let a=t.kernelShape;(a.length===0||a[0]===0)&&(a=[e.inputs[1].dims[2]]);let n=t.dilations;(n.length===0||n[0]===0)&&(n=[1]);let s=t.strides;(s.length===0||s[0]===0)&&(s=[1]);let u=t.pads;u.length===0&&(u=[0,0]),u=[0,u[0],0,u[1]],s=[1].concat(s),n=[1].concat(n),a=[1].concat(a);let d=t.outputPadding;d=[0].concat(d);let p=Vi({...t,pads:u,strides:s,dilations:n,kernelShape:a,outputPadding:d},i);Gi(e,i,p,c=>r?[c[0],c[2],c[3]]:[c[0],c[1],c[3]])},Kc=(e,t)=>{if(Pu(e.inputs,t),e.inputs[0].dims.length===3)qu(e,t);else{let r=Vi(t,e.inputs);Gi(e,e.inputs,r)}}}),Wu,Zc,Qc,h0=P(()=>{J(),ie(),ve(),ae(),Wu=(e,t,r,i)=>{let a=B.size(t),n=t.length,s=M("input",e,n),u=j("output",e,n),d=r.dataType===6?r.getInt32Array()[0]:Number(r.getBigInt64Array()[0]),p=B.normalizeAxis(d,n),c=f=>{let g=` i32(${s.indicesGet("inputIndices","uniforms.axis")}) `,_=Z("uniforms.input_shape","uniforms.axis",n),w=i.reverse?g+(i.exclusive?" + 1":""):"0",b=i.reverse?_:g+(i.exclusive?"":" + 1");return`
                ${f.registerUniform("outputSize","u32").registerUniform("axis","u32").declareVariables(s,u)}
                ${f.mainStart()}
                  ${f.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
                  var inputIndices = ${u.offsetToIndices("global_idx")};
                  var sum = ${u.type.value}(0);
                  let first : i32 = ${w};
                  let last : i32 = ${b};
                  for (var i : i32 = first; i < last; i++) {
                    ${s.indicesSet("inputIndices","uniforms.axis","u32(i)")};
                    sum = sum + ${s.getByIndices("inputIndices")};
                  }
                  ${u.setByOffset("global_idx","sum")};
                }`};return{name:"CumSum",shaderCache:{hint:i.cacheKey,inputDependencies:["rank"]},getRunData:()=>({outputs:[{dims:t,dataType:e}],dispatchGroup:{x:Math.ceil(a/64)},programUniforms:[{type:12,data:a},{type:12,data:p},...Q(t,t)]}),getShaderSource:c}},Zc=(e,t)=>{let r=e.inputs[0].dims,i=e.inputs[0].dataType,a=e.inputs[1];e.compute(Wu(i,r,a,t),{inputs:[0]})},Qc=e=>{let t=e.exclusive===1,r=e.reverse===1;return ce({exclusive:t,reverse:r})}}),Lu,Vu,Gu,Yc,Xc,f0=P(()=>{J(),ie(),ve(),ae(),Lu=e=>{if(!e||e.length!==1)throw new Error("DepthToSpace requires 1 input.");if(e[0].dims.length!==4)throw new Error("DepthToSpace requires 4D input.")},Vu=(e,t,r,i)=>{let a=[];a.push(`fn perm(i: ${i.type.indices}) -> ${r.type.indices} {
    var a: ${r.type.indices};`);for(let n=0;n<t;++n)a.push(r.indicesSet("a",e[n],`i[${n}]`));return a.push("return a;}"),a.join(`
`)},Gu=(e,t)=>{let r,i,a,n,s,u,d=t.format==="NHWC",p=t.blocksize,c=t.mode==="DCR";d?([r,i,a,n]=e.dims,s=c?[r,i,a,p,p,n/p**2]:[r,i,a,n/p**2,p,p],u=c?[0,1,3,2,4,5]:[0,1,4,2,5,3]):([r,i,a,n]=[e.dims[0],e.dims[2],e.dims[3],e.dims[1]],s=c?[r,p,p,n/p**2,i,a]:[r,n/p**2,p,p,i,a],u=c?[0,3,4,1,5,2]:[0,1,4,2,5,3]);let f=e.reshape(s),g=f.dims.length,_=e.dataType,w=M("a",_,g),b=j("output",_,g),S=v=>`
  ${v.registerUniform("output_size","u32").declareVariables(w,b)}

  ${Vu(u,g,w,b)}

  ${v.mainStart()}
    ${v.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}

    let indices = ${b.offsetToIndices("global_idx")};
    let aIndices = perm(indices);

    ${b.setByOffset("global_idx",w.getByIndices("aIndices"))}
  }`;return{name:"DepthToSpace",shaderCache:{hint:`${e.dims};${t.blocksize};${t.mode}`,inputDependencies:["rank"]},getRunData:v=>{let $=d?[r,i*p,a*p,n/p**2]:[r,n/p**2,i*p,a*p],I=B.size($),T=f.dims,E=B.sortBasedOnPerm(T,u);return{outputs:[{dims:$,dataType:v[0].dataType}],dispatchGroup:{x:Math.ceil(I/64)},programUniforms:[{type:12,data:I},...Q(T,E)]}},getShaderSource:S}},Yc=(e,t)=>{Lu(e.inputs),e.compute(Gu(e.inputs[0],t))},Xc=e=>ce({blocksize:e.blocksize,mode:e.mode,format:e.format})}),Or,er,Hi,Hu,Fu,ju,Ku,Fi,Zu,Jc,eh,m0=P(()=>{J(),ie(),ve(),ae(),Or="[a-zA-Z]|\\.\\.\\.",er="("+Or+")+",Hi="^"+er+"$",Hu="("+er+",)*"+er,Fu="^"+Hu+"$",ju=class{constructor(e=-1){this.symbolToIndices=new Map,this.inputIndex=e}addSymbol(e,t){let r=this.symbolToIndices.get(e);r===void 0?r=[t]:r.push(t),this.symbolToIndices.set(e,r)}},Ku=class{constructor(e,t){var a;this.equation=t,this.hasEllipsis=!1,this.symbolToInfo=new Map,this.lhs=new Array,this.outputDims=[];let[r,i]=t.includes("->")?t.split("->",2):[t,""];if(!r.match(RegExp(Fu)))throw new Error("Invalid LHS term");if(r.split(",").forEach((n,s)=>{let u=e[s].dims.slice();if(!n.match(RegExp(Hi)))throw new Error("Invalid LHS term");let d=this.processTerm(n,!0,u,s);this.lhs.push(d)}),i==="")i+=[...this.symbolToInfo.entries()].filter(([n,s])=>s.count===1||n==="...").map(([n])=>n).join("");else if(!i.match(RegExp(er)))throw new Error("Invalid RHS");(a=i.match(RegExp(Or,"g")))==null||a.forEach(n=>{if(n==="...")this.outputDims=this.outputDims.concat(this.ellipsisDims);else{let s=this.symbolToInfo.get(n);if(s===void 0)throw new Error("Invalid RHS symbol");this.outputDims.push(s.dimValue)}}),this.rhs=this.processTerm(i,!1,this.outputDims)}addSymbol(e,t,r){let i=this.symbolToInfo.get(e);if(i!==void 0){if(i.dimValue!==t&&i.count!==1)throw new Error("Dimension mismatch");i.count++,i.inputIndices.push(r)}else i={count:1,dimValue:t,inputIndices:[r]};this.symbolToInfo.set(e,i)}processTerm(e,t,r,i=-1){let a=r.length,n=!1,s=[],u=0;if(!e.match(RegExp(Hi))&&!t&&e!=="")throw new Error("Invalid LHS term");let d=e.match(RegExp(Or,"g")),p=new ju(i);return d==null||d.forEach((c,f)=>{if(c==="..."){if(n)throw new Error("Only one ellipsis is allowed per input term");n=!0;let g=a-d.length+1;if(g<0)throw new Error("Ellipsis out of bounds");if(s=r.slice(u,u+g),this.hasEllipsis){if(this.ellipsisDims.length!==s.length||this.ellipsisDims.toString()!==s.toString())throw new Error("Ellipsis dimensions mismatch")}else if(t)this.hasEllipsis=!0,this.ellipsisDims=s;else throw new Error("Ellipsis must be specified in the LHS");for(let _=0;_<s.length;_++){let w=String.fromCharCode(48+_);p.addSymbol(w,f+_),this.addSymbol(w,r[u++],i)}}else p.addSymbol(c,f+(this.hasEllipsis?this.ellipsisDims.length-1:0)),this.addSymbol(c,r[u++],i)}),p}},Fi=e=>e+"_max",Zu=(e,t,r,i)=>{let a=e.map(p=>p.length).map((p,c)=>M(`input${c}`,t,p)),n=B.size(i),s=j("output",t,i.length),u=[...r.symbolToInfo.keys()].filter(p=>!r.rhs.symbolToIndices.has(p)),d=p=>{let c=[],f="var prod = 1.0;",g="var sum = 0.0;",_="sum += prod;",w=[],b=[],S=[],v=[],$=r.symbolToInfo.size===r.rhs.symbolToIndices.size;r.symbolToInfo.forEach((T,E)=>{var A;if(r.rhs.symbolToIndices.has(E)){let C=(A=r.rhs.symbolToIndices.get(E))==null?void 0:A[0];C!==void 0&&r.lhs.forEach((O,U)=>{if(T.inputIndices.includes(U)){let x=O.symbolToIndices.get(E);if(x===void 0)throw new Error("Invalid symbol error");x.forEach(Y=>{c.push(`${a[U].indicesSet(`input${U}Indices`,Y,s.indicesGet("outputIndices",C))}`)})}})}else r.lhs.forEach((C,O)=>{if(T.inputIndices.includes(O)){let U=C.symbolToIndices.get(E);if(U===void 0)throw new Error("Invalid symbol error");U.forEach(x=>{w.push(`${a[O].indicesSet(`input${O}Indices`,x,`${E}`)}`)}),v.push(`prod *= ${a[O].getByIndices(`input${O}Indices`)};`)}}),b.push(`for(var ${E}: u32 = 0; ${E} < uniforms.${Fi(E)}; ${E}++) {`),S.push("}")});let I=$?[...c,`let sum = ${a.map((T,E)=>T.getByIndices(`input${E}Indices`)).join(" * ")};`]:[...c,g,...b,...w,f,...v,_,...S];return`
            ${p.registerUniforms(u.map(T=>({name:`${Fi(T)}`,type:"u32"}))).registerUniform("outputSize","u32").declareVariables(...a,s)}

            ${p.mainStart()}
            ${p.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
            var outputIndices = ${s.offsetToIndices("global_idx")};
            ${a.map((T,E)=>`var input${E}Indices: ${a[E].type.indices};`).join(`
`)}
            ${I.join(`
`)};
            ${s.setByOffset("global_idx","sum")};
          }`};return{name:"Einsum",shaderCache:{hint:r.equation,inputDependencies:e.map(()=>"rank")},getRunData:()=>{let p=u.filter(f=>r.symbolToInfo.has(f)).map(f=>{var g;return{type:12,data:((g=r.symbolToInfo.get(f))==null?void 0:g.dimValue)||0}});p.push({type:12,data:n});let c=e.map((f,g)=>[...Q(f)]).reduce((f,g)=>f.concat(g),p);return c.push(...Q(i)),{outputs:[{dims:i,dataType:t}],dispatchGroup:{x:Math.ceil(n/64)},programUniforms:c}},getShaderSource:d}},Jc=(e,t)=>{let r=new Ku(e.inputs,t.equation),i=r.outputDims,a=e.inputs.map((n,s)=>n.dims);e.compute(Zu(a,e.inputs[0].dataType,r,i))},eh=e=>{let t=e.equation.replace(/\s+/g,"");return ce({equation:t})}}),Qu,ji,Yu,Xu,th,g0=P(()=>{J(),ie(),ae(),Qu=e=>{if(!e||e.length!==2)throw new Error("Expand requires 2 input.");let t=e[0].dims,r=Array.from(e[1].getBigInt64Array(),Number),i=r.length<t.length?0:r.length-t.length,a=t.length<r.length?0:t.length-r.length;for(;i<r.length&&a<t.length;++i,++a)if(r[i]!==t[a]&&r[i]!==1&&t[a]!==1)throw new Error("Expand requires shape to be broadcastable to input")},ji=(e,t)=>{let r=e.length-t.length,i=[];for(let a=0;a<r;++a)i.push(e[a]);for(let a=0;a<t.length;++a)i.push(t[a]===1?e[a+r]:t[a]);return i},Yu=(e,t)=>e.length>t.length?ji(e,t):ji(t,e),Xu=e=>{let t=e[0].dims,r=Array.from(e[1].getBigInt64Array(),Number),i=Yu(t,r),a=e[0].dataType,n=a===9||B.size(t)===1,s=a===9||t.length>0&&t[t.length-1]%4===0?4:1,u=n||i.length>0&&i[i.length-1]%4===0?4:1,d=Math.ceil(B.size(i)/u),p=f=>{let g=M("input",a,t.length,s),_=j("output",a,i.length,u),w;if(a===9){let b=(S,v,$="")=>`
          let outputIndices${v} = ${_.offsetToIndices(`outputOffset + ${v}u`)};
          let offset${v} = ${g.broadcastedIndicesToOffset(`outputIndices${v}`,_)};
          let index${v} = offset${v} / 4u;
          let component${v} = offset${v} % 4u;
          ${S}[${v}] = ${$}(${g.getByOffset(`index${v}`)}[component${v}]);
        `;w=`
        let outputOffset = global_idx * ${u};
        var data = vec4<u32>(0);
        ${b("data",0,"u32")}
        ${b("data",1,"u32")}
        ${b("data",2,"u32")}
        ${b("data",3,"u32")}
        ${_.setByOffset("global_idx","data")}
      }`}else w=`
        let outputIndices = ${_.offsetToIndices(`global_idx * ${u}`)};
        let inputOffset = ${g.broadcastedIndicesToOffset("outputIndices",_)};
        let data = ${_.type.value}(${g.getByOffset(`inputOffset / ${s}`)});
        ${_.setByOffset("global_idx","data")}
      }`;return`
    ${f.registerUniform("vec_size","u32").declareVariables(g,_)}
    ${f.mainStart()}
    ${f.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")}
    ${w}`},c=[{type:12,data:d},...Q(t,i)];return{name:"Expand",shaderCache:{hint:`${i.length};${s}${u}`,inputDependencies:["rank"]},getShaderSource:p,getRunData:()=>({outputs:[{dims:i,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(d/64)},programUniforms:c})}},th=e=>{Qu(e.inputs),e.compute(Xu(e.inputs),{inputs:[0]})}}),Ju,rh,y0=P(()=>{J(),ie(),ae(),Ga(),Ju=e=>{let t=e[0].dataType,r=B.size(e[0].dims),i=B.size(e[1].dims),a=i%4===0,n=s=>{let u=M("x",t,[1],4),d=M("bias",t,[1],4),p=j("y",t,[1],4),c=[{name:"output_vec_size",type:"u32"},{name:"bias_size",type:"u32"}],f=_=>`
      let bias${_}_offset: u32 = (global_idx * 4 + ${_}) % uniforms.bias_size;
      let bias${_} = ${d.getByOffset(`bias${_}_offset / 4`)}[bias${_}_offset % 4];`,g=a?`
      let bias = ${d.getByOffset("global_idx % (uniforms.bias_size / 4)")};`:`${f(0)}${f(1)}${f(2)}${f(3)}
      let bias = ${u.type.value}(bias0, bias1, bias2, bias3);`;return`${s.registerUniforms(c).declareVariables(u,d,p)}

    ${ba(Ce(t))}

    ${s.mainStart(Lt)}
      ${s.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_vec_size")}

      let x = ${u.getByOffset("global_idx")};
      ${g}
      let x_in = x + bias;
      ${p.setByOffset("global_idx",$a("x_in"))}
    }`};return{name:"FastGeluWithBias",shaderCache:{hint:`${a}`,inputDependencies:["type","type"]},getShaderSource:n,getRunData:s=>({outputs:[{dims:s[0].dims,dataType:s[0].dataType}],programUniforms:[{type:12,data:Math.ceil(r/4)},{type:12,data:i}],dispatchGroup:{x:Math.ceil(r/Lt/4)}})}},rh=e=>{e.inputs.length<2||B.size(e.inputs[1].dims)===0?vc(e):e.compute(Ju(e.inputs))}}),el,tl,ih,ah,_0=P(()=>{J(),ie(),ve(),ae(),el=e=>{if(!e||e.length!==2)throw new Error("Gather requires 2 inputs.")},tl=(e,t)=>{let r=e[0].dims,i=e[1].dims,a=r.length,n=B.normalizeAxis(t.axis,a),s=r.slice(0);s.splice(n,1,...i);let u=r[n],d=e[0].dataType===9?4:1,p=Math.ceil(B.size(s)/d),c=[{type:12,data:p},{type:6,data:u},{type:12,data:n},...Q(e[0].dims,e[1].dims,s)],f=g=>{let _=M("data",e[0].dataType,e[0].dims.length,d),w=M("inputIndices",e[1].dataType,e[1].dims.length),b=j("output",e[0].dataType,s.length,d),S=$=>{let I=i.length,T=`var indicesIndices${$}  = ${w.type.indices}(0);`;for(let E=0;E<I;E++)T+=`${I>1?`indicesIndices${$}[${E}]`:`indicesIndices${$}`} = ${s.length>1?`outputIndices${$}[uniforms.axis + ${E}]`:`outputIndices${$}`};`;T+=`
          var idx${$} = ${w.getByIndices(`indicesIndices${$}`)};
          if (idx${$} < 0) {
            idx${$} = idx${$} + uniforms.axisDimLimit;
          }
          var dataIndices${$} : ${_.type.indices};
        `;for(let E=0,A=0;E<a;E++)E===n?(T+=`${a>1?`dataIndices${$}[${E}]`:`dataIndices${$}`} = u32(idx${$});`,A+=I):(T+=`${a>1?`dataIndices${$}[${E}]`:`dataIndices${$}`} = ${s.length>1?`outputIndices${$}[${A}]`:`outputIndices${$}`};`,A++);return T},v;if(e[0].dataType===9){let $=(I,T,E="")=>`
          let outputIndices${T} = ${b.offsetToIndices(`outputOffset + ${T}u`)};
          ${S(T)};
          let offset${T} = ${_.indicesToOffset(`dataIndices${T}`)};
          let index${T} = offset${T} / 4u;
          let component${T} = offset${T} % 4u;
          ${I}[${T}] = ${E}(${_.getByOffset(`index${T}`)}[component${T}]);
        `;v=`
        let outputOffset = global_idx * ${d};
        var value = vec4<u32>(0);
        ${$("value",0,"u32")}
        ${$("value",1,"u32")}
        ${$("value",2,"u32")}
        ${$("value",3,"u32")}
        ${b.setByOffset("global_idx","value")}
      `}else v=`
      let outputIndices = ${b.offsetToIndices("global_idx")};
      ${S("")};
      let value = ${_.getByIndices("dataIndices")};
      ${b.setByOffset("global_idx","value")};
      `;return`
      ${g.registerUniform("outputSize","u32").registerUniform("axisDimLimit","i32").registerUniform("axis","u32").declareVariables(_,w,b)}
      ${g.mainStart()}
        ${g.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
        ${v}
      }`};return{name:"Gather",shaderCache:{hint:t.cacheKey,inputDependencies:["rank","rank"]},getRunData:()=>({outputs:[{dims:s,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(p/64)},programUniforms:c}),getShaderSource:f}},ih=e=>ce({axis:e.axis}),ah=(e,t)=>{let r=e.inputs;el(r),e.compute(tl(e.inputs,t))}}),rl,nh,sh,w0=P(()=>{J(),ie(),ae(),rl=(e,t,r,i,a,n,s,u,d)=>{let p=[{type:12,data:n},{type:12,data:i},{type:12,data:a},{type:12,data:r},{type:12,data:s},{type:12,data:u},{type:12,data:d}],c=[n];p.push(...Q(t.dims,c));let f=g=>{let _=M("indices_data",t.dataType,t.dims.length),w=j("input_slice_offsets_data",12,1,1),b=[_,w],S=[{name:"output_size",type:"u32"},{name:"batch_dims",type:"u32"},{name:"input_dims",type:"u32",length:a.length},{name:"sizes_from_slice_dims_data",type:"u32",length:r.length},{name:"num_slices_per_batch",type:"u32"},{name:"input_batch_stride",type:"u32"},{name:"num_slice_dims",type:"u32"}];return`
  ${g.registerUniforms(S).declareVariables(...b)}
  ${g.mainStart()}
    ${g.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    let batch_idx = global_idx / uniforms.num_slices_per_batch;
    let base_offset = batch_idx * uniforms.input_batch_stride;

    let slice_indices_base_offset = global_idx * uniforms.num_slice_dims;
    var relative_slice_offset = 0;
    for (var dim_idx = 0u; dim_idx < uniforms.num_slice_dims; dim_idx ++) {
      var index = i32(indices_data[dim_idx + slice_indices_base_offset].x);
      let input_dim_idx = uniforms.batch_dims + dim_idx;
      if (index < 0) {
        ${a.length===1?"index += i32(uniforms.input_dims);":"index += i32(uniforms.input_dims[input_dim_idx]);"}
      }
      ${r.length===1?"relative_slice_offset += index * i32(uniforms.sizes_from_slice_dims_data);":"relative_slice_offset += index * i32(uniforms.sizes_from_slice_dims_data[dim_idx]);"}
    }

    input_slice_offsets_data[global_idx] =  base_offset + u32(relative_slice_offset);
  }`};return e.compute({name:"computeSliceOffsets",shaderCache:{hint:`${a.length}_${r.length}`,inputDependencies:["rank"]},getRunData:()=>({outputs:[{dims:c,dataType:e.inputs[1].dataType}],dispatchGroup:{x:Math.ceil(n/64)},programUniforms:p}),getShaderSource:f},{inputs:[t],outputs:[-1]})[0]},nh=(e,t)=>{let r=e.inputs,i=r[0].dims,a=r[0].dataType,n=r[1].dims,s=n[n.length-1],u=B.sizeToDimension(n,n.length-1),d=B.sizeFromDimension(i,t.batchDims+s),p=B.sizeToDimension(i,t.batchDims),c=B.sizeFromDimension(i,t.batchDims),f=u/p,g=new Array(s),_=d;for(let T=0;T<s;++T)g[s-1-T]=_,_*=i[t.batchDims+s-1-T];let w=rl(e,r[1],g,t.batchDims,i,u,f,c,s),b=t.batchDims+s;if(b>i.length)throw new Error("last dimension of indices must not be larger than rank of input tensor");let S=n.slice(0,-1).concat(i.slice(b)),v=B.size(S),$=[{type:12,data:v},{type:12,data:d},...Q(r[0].dims,w.dims,S)],I=T=>{let E=M("data",r[0].dataType,r[0].dims.length),A=M("slice_offsets",12,w.dims.length),C=j("output",r[0].dataType,S.length);return`
          ${T.registerUniform("output_size","u32").registerUniform("slice_size","u32").declareVariables(E,A,C)}
            ${T.mainStart()}
            ${T.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
          let slice_offset = slice_offsets[global_idx / uniforms.slice_size];
          output[global_idx] = data[u32(slice_offset) + global_idx % uniforms.slice_size];
        }`};e.compute({name:"GatherND",shaderCache:{hint:t.cacheKey,inputDependencies:["rank","rank"]},getRunData:()=>({outputs:[{dims:S,dataType:a}],dispatchGroup:{x:Math.ceil(v/64)},programUniforms:$}),getShaderSource:I},{inputs:[r[0],w]})},sh=e=>({batchDims:e.batch_dims,cacheKey:""})}),il,al,oh,uh,b0=P(()=>{J(),ie(),ve(),ae(),il=(e,t)=>{if(e.length<3||e.length>4)throw new Error("GatherBlockQuantized requires 3 or 4 inputs.");let r=B.normalizeAxis(t.quantizeAxis,e[0].dims.length),i=t.blockSize,a=e[0],n=e[2],s=e.length===4?e[3]:void 0;if(n.dims.length!==a.dims.length||!a.dims.map((u,d)=>d===r?Math.ceil(u/i)===n.dims[d]:u===n.dims[d]).reduce((u,d)=>u&&d,!0))throw new Error("Scales must have the same rank as the input tensor and the dims should match except on gatherAxis.");if(s){if(s.dataType!==a.dataType)throw new Error("Zero point must have the same data type as the input tensor.");if(s.dims.length!==n.dims.length||!s.dims.map((u,d)=>u===n.dims[d]).reduce((u,d)=>u&&d,!0))throw new Error("Zero point must have the same rank as the input tensor and the dims should match except on quantizeAxis.")}},al=(e,t)=>{let r=e[0].dims,i=e[1].dims,a=r.length,n=B.normalizeAxis(t.gatherAxis,a),s=B.normalizeAxis(t.quantizeAxis,a),u=r.slice(0);u.splice(n,1,...i);let d=B.size(u),p=e[2].dataType,c=e[0].dataType===22,f=[{type:12,data:d},{type:12,data:s},{type:12,data:n},{type:12,data:t.blockSize},...Q(...e.map((_,w)=>_.dims),u)],g=_=>{let w=M("data",e[0].dataType,e[0].dims.length),b=M("inputIndices",e[1].dataType,e[1].dims.length),S=M("scales",e[2].dataType,e[2].dims.length),v=e.length>3?M("zeroPoint",e[3].dataType,e[3].dims.length):void 0,$=j("output",p,u.length),I=[w,b,S];v&&I.push(v);let T=[{name:"output_size",type:"u32"},{name:"quantize_axis",type:"u32"},{name:"gather_axis",type:"u32"},{name:"block_size",type:"u32"}];return`
        ${_.registerUniforms(T).declareVariables(...I,$)}
        ${_.mainStart()}
        let output_indices = ${$.offsetToIndices("global_idx")};
        var indices_indices = ${b.type.indices}(0);
        ${i.length>1?`
          for (var i: u32 = 0; i < ${i.length}; i++) {
            let index = ${$.indicesGet("output_indices","uniforms.gather_axis + i")};
            ${b.indicesSet("indices_indices","i","index")};
          }`:`indices_indices = ${$.indicesGet("output_indices","uniforms.gather_axis")};`};
        var data_indices = ${w.type.indices}(0);
        for (var i: u32 = 0; i < uniforms.gather_axis; i++) {
          let index = ${$.indicesGet("output_indices","i")};
          ${w.indicesSet("data_indices","i","index")};
        }
        var index_from_indices = ${b.getByIndices("indices_indices")};
        if (index_from_indices < 0) {
          index_from_indices += ${r[n]};
        }
        ${w.indicesSet("data_indices","uniforms.gather_axis","u32(index_from_indices)")};
        for (var i = uniforms.gather_axis + 1; i < ${u.length}; i++) {
          let index = ${$.indicesGet("output_indices",`i + ${i.length} - 1`)};
          ${w.indicesSet("data_indices","i","index")};
        }
        let data_offset = ${w.indicesToOffset("data_indices")};
        let data_index = data_offset % 8;
        // Convert 4-bit packed data to 8-bit packed data.
        let packed_4bit_quantized_data = ${w.getByOffset("data_offset / 8")};
        let packed_8bit_quantized_data = (packed_4bit_quantized_data >> (4 * (data_index % 2))) & 0x0f0f0f0f;
        let quantized_data_vec = ${c?"unpack4xI8":"unpack4xU8"}(u32(packed_8bit_quantized_data));
        let quantized_data = quantized_data_vec[data_index / 2];
        var scale_indices = data_indices;
        let quantize_axis_index = ${S.indicesGet("data_indices","uniforms.quantize_axis")} / uniforms.block_size;
        ${S.indicesSet("scale_indices","uniforms.quantize_axis","quantize_axis_index")};
        var scale = ${S.getByIndices("scale_indices")};
        ${v?`
              let zero_point_indices = scale_indices;
              let zero_point_offset = ${v.indicesToOffset("zero_point_indices")};
              let zero_point_index = zero_point_offset % 8;
              let packed_4bit_zero_points = ${v.getByOffset("zero_point_offset / 8")};
              let packed_8bit_zero_points = (packed_4bit_zero_points >> (4 * (zero_point_index % 2))) & 0x0f0f0f0f;
              let zero_point_vec = ${c?"unpack4xI8":"unpack4xU8"}(u32(packed_8bit_zero_points));
              let zero_point = zero_point_vec[zero_point_index / 2];`:"var zero_point = 0"};
        let dequantized_data = ${Ce(p)}(quantized_data - zero_point) * scale;
        ${$.setByOffset("global_idx","dequantized_data")};
    }`};return{name:"GatherBlockQuantized",shaderCache:{hint:`${t.cacheKey};${e.filter((_,w)=>w!==1).map(_=>_.dims.join("_")).join(";")}`,inputDependencies:Array.from({length:e.length},(_,w)=>"rank")},getRunData:()=>({outputs:[{dims:u,dataType:p}],dispatchGroup:{x:Math.ceil(d/64)},programUniforms:f}),getShaderSource:g}},oh=(e,t)=>{let r=e.inputs;il(r,t),e.compute(al(e.inputs,t))},uh=e=>ce({blockSize:e.blockSize,gatherAxis:e.gatherAxis,quantizeAxis:e.quantizeAxis})}),nl,sl,lh,dh,$0=P(()=>{J(),ie(),ve(),ae(),nl=e=>{if(!e||e.length!==2)throw new Error("GatherElements requires 2 inputs.");if(e[0].dims.length<1)throw new Error("GatherElements requires that the data input be rank >= 1.");if(e[0].dims.length!==e[1].dims.length)throw new Error(`GatherElements requires that the data input and
                     indices input tensors be of same rank.`)},sl=(e,t)=>{let r=e[0].dims,i=e[0].dataType,a=r.length,n=e[1].dims,s=e[1].dataType,u=B.normalizeAxis(t.axis,a),d=r[u],p=n.slice(0),c=B.size(p),f=M("input",i,a),g=M("indicesInput",s,n.length),_=j("output",i,p.length),w=[{type:12,data:c},{type:6,data:d},{type:12,data:u}];return w.push(...Q(r,n,p)),{name:"GatherElements",shaderCache:{inputDependencies:["rank","rank"]},getRunData:()=>({outputs:[{dims:p,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(c/64)},programUniforms:w}),getShaderSource:b=>`
      ${b.registerUniform("outputSize","u32").registerUniform("axisDimLimit","i32").registerUniform("axis","u32").declareVariables(f,g,_)}
      ${b.mainStart()}
      ${b.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}

      let outputIndices = ${_.offsetToIndices("global_idx")};

      var idx = ${g.getByOffset("global_idx")};
      if (idx < 0) {
        idx = idx + uniforms.axisDimLimit;
      }
      var inputIndices = ${f.type.indices}(outputIndices);
      ${f.indicesSet("inputIndices","uniforms.axis","u32(idx)")};
      let value = ${f.getByIndices("inputIndices")};

      ${_.setByOffset("global_idx","value")};
  }`}},lh=e=>ce({axis:e.axis}),dh=(e,t)=>{let r=e.inputs;nl(r),e.compute(sl(e.inputs,t))}}),ol,ul,ph,ch,v0=P(()=>{J(),ie(),ae(),ol=e=>{if(!e)throw new Error("Input is missing");if(e.length<2||e.length>3)throw new Error("Invaid input number.");if(e.length===3&&e[2].dims.length>2)throw new Error("Invalid input shape of C");if(e[0].dataType!==e[1].dataType||e.length===3&&e[0].dataType!==e[2].dataType)throw new Error("Input types are mismatched")},ul=(e,t)=>{let r=e[0].dims.slice(),i=e[1].dims.slice(),[a,n,s]=lp.getShapeOfGemmResult(r,t.transA,i,t.transB,e.length===3?e[2].dims:void 0),u=[a,n];if(!u)throw new Error("Can't use gemm on the given tensors");let d=16,p=Math.ceil(n/d),c=Math.ceil(a/d),f=!0,g=B.size(u),_=[{type:12,data:f?p:g},{type:12,data:a},{type:12,data:n},{type:12,data:s},{type:1,data:t.alpha},{type:1,data:t.beta}],w=["type","type"];e.length===3&&(_.push(...Q(e[2].dims)),w.push("rank")),_.push(...Q(u));let b=v=>{let $="";t.transA&&t.transB?$="value += a[k * uniforms.M + m] * b[n * uniforms.K + k];":t.transA&&!t.transB?$="value += a[k * uniforms.M + m] * b[k * uniforms.N + n];":!t.transA&&t.transB?$="value += a[m * uniforms.K + k] * b[n * uniforms.K + k];":!t.transA&&!t.transB&&($="value += a[m * uniforms.K + k] * b[k * uniforms.N + n];");let I=t.alpha===1?"":"value *= uniforms.alpha;",T=M("a",e[0].dataType,e[0].dims),E=M("b",e[1].dataType,e[1].dims),A=T.type.value,C=null,O=[T,E];e.length===3&&(C=M("c",e[2].dataType,e[2].dims.length),O.push(C));let U=j("output",e[0].dataType,u.length);O.push(U);let x=[{name:"output_size",type:"u32"},{name:"M",type:"u32"},{name:"N",type:"u32"},{name:"K",type:"u32"},{name:"alpha",type:"f32"},{name:"beta",type:"f32"}];return`
  ${v.registerUniforms(x).declareVariables(...O)}

  ${v.mainStart()}
    ${v.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}

    let m = global_idx / uniforms.N;
    let n = global_idx % uniforms.N;

    var value = ${A}(0);
    for (var k: u32 = 0u; k < uniforms.K; k++) {
      ${$}
    }

    ${I}
    ${C!=null?`let cOffset = ${C.broadcastedIndicesToOffset("vec2(m, n)",U)}; value += ${A}(uniforms.beta) * ${C.getByOffset("cOffset")};`:""}
    output[global_idx] = value;
  }`},S=v=>{let $=M("a",e[0].dataType,e[0].dims),I=M("b",e[1].dataType,e[1].dims),T=null,E=[$,I];e.length===3&&(T=M("c",e[2].dataType,e[2].dims.length),E.push(T));let A=j("output",e[0].dataType,u.length);E.push(A);let C=[{name:"num_tile_n",type:"u32"},{name:"M",type:"u32"},{name:"N",type:"u32"},{name:"K",type:"u32"},{name:"alpha",type:"f32"},{name:"beta",type:"f32"}],O="",U="";t.transA&&t.transB?(U=`
      var col = tile_row_start + local_id.x;
      var row = k_start + local_id.y;
      if (col < uniforms.M && row < uniforms.K) {
        tile_a[local_id.y][local_id.x] = a[row * uniforms.M + col];
      } else {
        tile_a[local_id.y][local_id.x] = ${$.type.value}(0);
      }

      col = k_start + local_id.x;
      row = tile_col_start + local_id.y;
      if (col < uniforms.K && row < uniforms.N) {
        tile_b[local_id.y][local_id.x] = b[row * uniforms.K + col];
      } else {
        tile_b[local_id.y][local_id.x] = ${I.type.value}(0);
      }
      `,O="value += tile_a[k][local_id.y] * tile_b[local_id.x][k];"):t.transA&&!t.transB?(U=`
      var col = tile_row_start + local_id.x;
      var row = k_start + local_id.y;
      if (col < uniforms.M && row < uniforms.K) {
        tile_a[local_id.y][local_id.x] = a[row * uniforms.M + col];
      } else {
        tile_a[local_id.y][local_id.x] = ${$.type.value}(0);
      }

      col = tile_col_start + local_id.x;
      row = k_start + local_id.y;
      if (col < uniforms.N && row < uniforms.K) {
        tile_b[local_id.y][local_id.x] = b[row * uniforms.N + col];
      } else {
        tile_b[local_id.y][local_id.x] = ${I.type.value}(0);
      }
      `,O="value += tile_a[k][local_id.y] * tile_b[k][local_id.x];"):!t.transA&&t.transB?(U=`
      var col = k_start + local_id.x;
      var row = tile_row_start + local_id.y;
      if (col < uniforms.K && row < uniforms.M) {
        tile_a[local_id.y][local_id.x] = a[row * uniforms.K + col];
      } else {
        tile_a[local_id.y][local_id.x] = ${$.type.value}(0);
      }

      col = k_start + local_id.x;
      row = tile_col_start + local_id.y;
      if (col < uniforms.K && row < uniforms.N) {
        tile_b[local_id.y][local_id.x] = b[row * uniforms.K + col];
      } else {
        tile_b[local_id.y][local_id.x] = ${I.type.value}(0);
      }
      `,O="value += tile_a[local_id.y][k] * tile_b[local_id.x][k];"):!t.transA&&!t.transB&&(U=`
      var col = k_start + local_id.x;
      var row = tile_row_start + local_id.y;
      if (col < uniforms.K && row < uniforms.M) {
        tile_a[local_id.y][local_id.x] = a[row * uniforms.K + col];
      } else {
        tile_a[local_id.y][local_id.x] = ${$.type.value}(0);
      }

      col = tile_col_start + local_id.x;
      row = k_start + local_id.y;
      if (col < uniforms.N && row < uniforms.K) {
        tile_b[local_id.y][local_id.x] = b[row * uniforms.N + col];
      } else {
        tile_b[local_id.y][local_id.x] = ${I.type.value}(0);
      }
      `,O="value += tile_a[local_id.y][k] * tile_b[k][local_id.x];");let x=t.alpha===1?"":"value *= uniforms.alpha;";return`
  ${v.registerUniforms(C).declareVariables(...E)}
  var<workgroup> tile_a: array<array<${$.type.storage}, ${d}>, ${d}>;
  var<workgroup> tile_b: array<array<${I.type.storage}, ${d}>, ${d}>;
  ${v.mainStart([d,d,1])}
    let tile_col_start = (workgroup_index % uniforms.num_tile_n) * ${d};
    let tile_row_start = (workgroup_index / uniforms.num_tile_n) * ${d};
    let num_tiles = (uniforms.K - 1) / ${d} + 1;
    var k_start = 0u;
    var value = ${A.type.value}(0);
    for (var t: u32 = 0u; t < num_tiles; t++) {
      ${U}
      k_start = k_start + ${d};
      workgroupBarrier();

      for (var k: u32 = 0u; k < ${d}; k++) {
        ${O}
      }
      workgroupBarrier();
    }

    ${x}
    let m = tile_row_start + local_id.y;
    let n = tile_col_start + local_id.x;
    ${T!=null?`let cOffset = ${T.broadcastedIndicesToOffset("vec2(m, n)",A)}; value += ${A.type.value}(uniforms.beta) * ${T.getByOffset("cOffset")};`:""}
    if (m < uniforms.M && n < uniforms.N) {
      output[m * uniforms.N + n] = value;
    }
  }`};return f?{name:"GemmShared",shaderCache:{hint:`${t.cacheKey}`,inputDependencies:w},getRunData:()=>({outputs:[{dims:u,dataType:e[0].dataType}],dispatchGroup:{x:p*c},programUniforms:_}),getShaderSource:S}:{name:"Gemm",shaderCache:{hint:`${t.cacheKey}`,inputDependencies:w},getRunData:()=>({outputs:[{dims:u,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(g/64)},programUniforms:_}),getShaderSource:b}},ph=e=>{let t=e.transA,r=e.transB,i=e.alpha,a=e.beta;return{transA:t,transB:r,alpha:i,beta:a,cacheKey:`${e.transA};${e.transB};${e.alpha===1}`}},ch=(e,t)=>{ol(e.inputs),e.compute(ul(e.inputs,t))}}),Je,it,vt,xt,ll,dl,pl,cl,hl,fl,ml,gl,hh,fh,x0=P(()=>{J(),ie(),ve(),ae(),[Je,it,vt,xt]=[0,1,2,3],ll=e=>{if(e[0].dims.length!==4)throw new Error("only 4-D tensor is supported.");if(e[0].dims.length!==e[1].dims.length)throw new Error("input dimensions must be equal to grid dimensions");if(e[0].dims.length-2!==e[1].dims[e[1].dims.length-1])throw new Error(`last dimension of grid must be equal to ${e[0].dims.length-2}`);if(e[0].dims[0]!==e[1].dims[0])throw new Error("grid batch size must match input batch size")},dl=`
  fn gs_get_cubic_coeffs(x: f32) -> vec4<f32> {
    let cubic_alpha = -0.75f;
    let x_abs = abs(x);
    var coeffs: vec4<f32>;
    coeffs[0] = (((cubic_alpha * (x_abs + 1) - 5 * cubic_alpha) * (x_abs + 1) + 8 * cubic_alpha) * (x_abs + 1) - 4 * cubic_alpha);
    coeffs[1] = (((cubic_alpha + 2) * x_abs - (cubic_alpha + 3)) * x_abs * x_abs + 1);
    coeffs[2] = (((cubic_alpha + 2) * (1 - x_abs) - (cubic_alpha + 3)) * (1 - x_abs) * (1 - x_abs) + 1);
    coeffs[3] = (((cubic_alpha * (2 - x_abs) - 5 * cubic_alpha) * (2 - x_abs) + 8 * cubic_alpha) * (2 - x_abs) - 4 * cubic_alpha);
    return coeffs;
  }
`,pl=e=>`
  fn gs_bicubic_interpolate(p: mat4x4<${e}>, x: f32, y: f32) -> ${e} {
    var v: vec4<f32>;
    var coeffs = gs_get_cubic_coeffs(x);
    for (var i = 0; i < 4; i++) {
      v[i] = coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
    }
    coeffs = gs_get_cubic_coeffs(y);
    let pixel = ${e}(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
    return pixel;
  }
`,cl=e=>`
  fn gs_denormalize(n: f32, length: i32) -> f32 {
    ${e.alignCorners===0?`
    // alignCorners: false => [-1, 1] to [-0.5, length - 0.5]
    return ((n + 1.0) * f32(length) - 1.0) / 2.0;
    `:`
    // alignCorners: true => [-1, 1] to [0, length - 1]
    return (n + 1.0) / 2.0 * (f32(length - 1));
    `}
  }
`,hl=e=>`
  ${e.paddingMode==="reflection"?`
      fn gs_reflect(x: i32, x_min: f32, x_max: f32) -> u32 {
        var dx = 0.0;
        var fx = f32(x);
        let range = x_max - x_min;
        if (fx < x_min) {
          dx = x_min - fx;
          let n = u32(dx / range);
          let r = dx - f32(n) * range;
          if (n % 2 == 0) {
            fx = x_min + r;
          } else {
            fx = x_max - r;
          }
        } else if (fx > x_max) {
          dx = fx - x_max;
          let n = u32(dx / range);
          let r = dx - f32(n) * range;
          if (n % 2 == 0) {
            fx = x_max - r;
          } else {
            fx = x_min + r;
          }
        }
        return u32(fx);
      }`:""}
`,fl=(e,t,r)=>`
  fn pixel_at_grid(r: i32, c: i32, H: i32, W: i32, batch: u32, channel: u32, border: vec4<f32>) -> ${t} {
     var pixel = ${t}(0);
     var indices = vec4<u32>(0);
     indices[${Je}] = batch;
     indices[${it}] = channel;`+(()=>{switch(r.paddingMode){case"zeros":return`
          if (r >= 0 && r < H && c >=0 && c < W) {
            indices[${vt}] = u32(r);
            indices[${xt}] = u32(c);
          } else {
            return ${t}(0);
          }
        `;case"border":return`
          indices[${vt}] = u32(clamp(r, 0, H - 1));
          indices[${xt}] = u32(clamp(c, 0, W - 1));
        `;case"reflection":return`
          indices[${vt}] = gs_reflect(r, border[1], border[3]);
          indices[${xt}] = gs_reflect(c, border[0], border[2]);
        `;default:throw new Error(`padding mode ${r.paddingMode} is not supported`)}})()+`
    return ${e.getByIndices("indices")};
  }
`,ml=(e,t,r)=>(()=>{switch(r.mode){case"nearest":return`
          let result = pixel_at_grid(i32(round(y)), i32(round(x)), H_in, W_in, indices[${Je}], indices[${it}], border);
        `;case"bilinear":return`
          let x1 = i32(floor(x));
          let y1 = i32(floor(y));
          let x2 = x1 + 1;
          let y2 = y1 + 1;

          let p11 = pixel_at_grid(y1, x1, H_in, W_in, indices[${Je}], indices[${it}], border);
          let p12 = pixel_at_grid(y1, x2, H_in, W_in, indices[${Je}], indices[${it}], border);
          let p21 = pixel_at_grid(y2, x1, H_in, W_in, indices[${Je}], indices[${it}], border);
          let p22 = pixel_at_grid(y2, x2, H_in, W_in, indices[${Je}], indices[${it}], border);

          let dx2 = ${t}(f32(x2) - x);
          let dx1 = ${t}(x - f32(x1));
          let dy2 = ${t}(f32(y2) - y);
          let dy1 = ${t}(y - f32(y1));
          let result = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);
        `;case"bicubic":return`
          let x0 = i32(floor(x)) - 1;
          let y0 = i32(floor(y)) - 1;
          var p: mat4x4<${t}>;
          for (var h = 0; h < 4; h++) {
            for (var w = 0; w < 4; w++) {
              p[h][w] = pixel_at_grid(h + y0, w + x0, H_in, W_in, indices[${Je}], indices[${it}], border);
            }
          }

          let dx = x - f32(x0 + 1);
          let dy = y - f32(y0 + 1);
          let result = gs_bicubic_interpolate(p, dx, dy);
        `;default:throw new Error(`mode ${r.mode} is not supported`)}})()+`${e.setByOffset("global_idx","result")}`,gl=(e,t)=>{let r=M("x",e[0].dataType,e[0].dims.length),i=[e[1].dims[0],e[1].dims[1],e[1].dims[2]],a=M("grid",e[1].dataType,i.length,2),n=[e[0].dims[0],e[0].dims[1],e[1].dims[1],e[1].dims[2]];t.format==="NHWC"&&(n=[e[0].dims[0],e[1].dims[1],e[1].dims[2],e[0].dims[3]],[Je,it,vt,xt]=[0,3,1,2]);let s=j("output",e[0].dataType,n.length),u=r.type.value,d=B.size(n),p=[{type:12,data:d},...Q(e[0].dims,i,n)],c=f=>`
  ${f.registerUniform("output_size","u32").declareVariables(r,a,s)}
  ${dl}
  ${pl(u)}
  ${cl(t)}
  ${hl(t)}
  ${fl(r,u,t)}

  ${f.mainStart()}
    ${f.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
      let H_in = i32(uniforms.x_shape[${vt}]);
      let W_in = i32(uniforms.x_shape[${xt}]);

      ${t.alignCorners===0?`
      let x_min = -0.5;
      let x_max = f32(W_in) - 0.5;
      let y_min = -0.5;
      let y_max = f32(H_in) - 0.5;
      `:`
      let x_min = 0.0;
      let x_max = f32(W_in) - 1.0;
      let y_min = 0.0;
      let y_max = f32(H_in) - 1.0;
      `};
      let border = vec4<f32>(x_min, y_min, x_max, y_max);

      let indices = ${s.offsetToIndices("global_idx")};
      var grid_indices = vec3<u32>(indices[${Je}], indices[${vt}], indices[${xt}]);
      let nxy = ${a.getByIndices("grid_indices")};
      var x = gs_denormalize(f32(nxy[0]), W_in);
      var y = gs_denormalize(f32(nxy[1]), H_in);

      ${ml(s,u,t)}
  }`;return{name:"GridSample",shaderCache:{hint:`${t.cacheKey}`,inputDependencies:["type","type"]},getRunData:f=>{let g=B.size(n);return{outputs:[{dims:n,dataType:f[0].dataType}],dispatchGroup:{x:Math.ceil(g/64)},programUniforms:p}},getShaderSource:c}},hh=(e,t)=>{ll(e.inputs),e.compute(gl(e.inputs,t))},fh=e=>ce({alignCorners:e.align_corners,mode:e.mode,paddingMode:e.padding_mode,format:e.format})}),Ae,yl,mh,Ki,_l,ur,gh,yh=P(()=>{J(),ie(),ve(),qa(),Va(),ae(),yt(),Ae=(e,t)=>e.length>t&&e[t].dims.length>0?e[t]:void 0,yl=(e,t)=>{let r=e[0],i=Ae(e,1),a=Ae(e,2),n=Ae(e,3),s=Ae(e,4),u=Ae(e,5),d=Ae(e,6),p=Ae(e,7);if(r.dims.length!==3&&r.dims.length!==5)throw new Error("Input query is expected to have 3 or 5 dimensions");let c=r.dims[0],f=r.dims[1],g=r.dims.length===3?r.dims[2]:t.numHeads*r.dims[4],_=f,w=0,b=0,S=Math.floor(g/t.numHeads);if(d&&p&&B.size(d.dims)&&B.size(p.dims)){if(d.dims.length!==4)throw new Error('Input "past_key" is expected to have 4 dimensions');if(d.dims[0]!==c||d.dims[1]!==t.numHeads||d.dims[3]!==S)throw new Error('Input "past_key" shape (batch_size, num_heads, past_sequence_length, head_size)');if(p.dims[0]!==c||p.dims[1]!==t.numHeads||p.dims[3]!==S)throw new Error('Input "past_value" shape (batch_size, num_heads, past_sequence_length, head_size)');if(d.dims[2]!==p.dims[2])throw new Error('Input "past_key" and "past_value" shall have same dim 2 (past_sequence_length)');if(p.dims.length!==4)throw new Error('Input "past_value" is expected to have 4 dimensions');w=d.dims[2],b=d.dims[2]}else if(d&&B.size(d.dims)||p&&B.size(p.dims))throw new Error('Input "past_key" and "past_value" shall be both present or both absent');let v;if(i&&B.size(i.dims)>0){if(r.dims.length!==3)throw new Error('Input "query" is expected to have 3 dimensions when key is given');if(i.dims.length<3||i.dims.length>5)throw new Error('Input "key" is expected to have 3, 4, or 5 dimensions');if(r.dims[0]!==i.dims[0])throw new Error('Input "query" and "key" shall have same dim 0 (batch size)');if(i.dims.length===3){if(i.dims[2]!==r.dims[2])throw new Error('Input "query" and "key" shall have same dim 2 (hidden_size)');v=2,_=i.dims[1]}else if(i.dims.length===5){if(i.dims[2]!==t.numHeads||i.dims[3]!==2||i.dims[4]!==S)throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');if(a)throw new Error('Expect "value" be none when "key" has packed kv format.');v=5,_=i.dims[1]}else{if(i.dims[1]!==t.numHeads||i.dims[3]!==S)throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');v=0,_=i.dims[2]}}else{if(r.dims.length!==5)throw new Error('Input "query" is expected to have 5 dimensions when key is empty');if(r.dims[2]!==t.numHeads||r.dims[3]!==3)throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');v=3}if(n&&B.size(n.dims)>0){if(n.dims.length!==1)throw new Error('Input "bias" is expected to have 1 dimension');if(i&&i.dims.length===5&&i.dims[3]===2)throw new Error("bias is not allowed for packed kv.")}let $=w+_,I=0;if(s&&B.size(s.dims)>0){I=8;let C=s.dims;throw C.length===1?C[0]===c?I=1:C[0]===3*c+2&&(I=3):C.length===2&&C[0]===c&&C[1]===$&&(I=5),I===8?new Error('Input "key_padding_mask" shape shall be (batch_size) or (batch_size, total_sequence_length)'):new Error("Mask not supported")}let T=!1,E=g;if(a&&B.size(a.dims)>0){if(a.dims.length!==3&&a.dims.length!==4)throw new Error('Input "value" is expected to have 3 or 4 dimensions');if(r.dims[0]!==a.dims[0])throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');if(a.dims.length===3){if(_!==a.dims[1])throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');E=a.dims[2]}else{if(_!==a.dims[2])throw new Error('Input "key" and "value" shall have the same dim 2 (kv_sequence_length)');E=a.dims[1]*a.dims[3],T=!0}}let A=!1;if(s&&B.size(s.dims)>0)throw new Error("Key padding mask is not supported");if(u&&B.size(u.dims)>0){if(u.dims.length!==4)throw new Error('Input "attention_bias" is expected to have 4 dimensions');if(u.dims[0]!==c||u.dims[1]!==t.numHeads||u.dims[2]!==f||u.dims[3]!==$)throw new Error('Expect "attention_bias" shape (batch_size, num_heads, sequence_length, total_sequence_length)')}return{batchSize:c,sequenceLength:f,pastSequenceLength:w,kvSequenceLength:_,totalSequenceLength:$,maxSequenceLength:b,inputHiddenSize:0,hiddenSize:g,vHiddenSize:E,headSize:S,vHeadSize:Math.floor(E/t.numHeads),numHeads:t.numHeads,isUnidirectional:!1,pastPresentShareBuffer:!1,maskFilterValue:t.maskFilterValue,maskType:I,scale:t.scale,broadcastResPosBias:A,passPastInKv:T,qkvFormat:v}},mh=e=>ce({...e}),Ki=ce({perm:[0,2,1,3]}),_l=(e,t,r,i,a,n,s)=>{let u=[i,a,n],d=B.size(u),p=[{type:12,data:d},{type:12,data:s},{type:12,data:n}],c=f=>{let g=j("qkv_with_bias",t.dataType,u),_=M("qkv",t.dataType,u),w=M("bias",r.dataType,u),b=[{name:"output_size",type:"u32"},{name:"bias_offset",type:"u32"},{name:"hidden_size",type:"u32"}];return`
  ${f.registerUniforms(b).declareVariables(_,w,g)}
  ${f.mainStart()}
    ${f.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    let bias_offset_idx = (global_idx % uniforms.hidden_size) + uniforms.bias_offset;

    qkv_with_bias[global_idx] = qkv[global_idx] + bias[bias_offset_idx];
  }`};return e.compute({name:"MultiHeadAttentionAddBias",shaderCache:{inputDependencies:["type","type"]},getRunData:()=>({outputs:[{dims:u,dataType:t.dataType,gpuDataType:0}],dispatchGroup:{x:Math.ceil(d/64)},programUniforms:p}),getShaderSource:c},{inputs:[t,r],outputs:[-1]})[0]},ur=(e,t,r,i,a,n,s,u)=>{let d=n;if(s&&B.size(s.dims)>0){if(i===1)throw new Error("AddBiasReshape is not implemented. Please export your model with packed QKV or KV");return d=_l(e,n,s,t,i,r*a,u),d=d.reshape([t,i,r,a]),r===1||i===1?d:e.compute(De(d,Ki.perm),{inputs:[d],outputs:[-1]})[0]}else return n.dims.length===3&&(d=n.reshape([t,i,r,a])),r===1||i===1?d:e.compute(De(d,Ki.perm),{inputs:[d],outputs:[-1]})[0]},gh=(e,t)=>{let r=yl(e.inputs,t),i=e.inputs[0],a=Ae(e.inputs,1),n=Ae(e.inputs,2),s=Ae(e.inputs,3),u=Ae(e.inputs,4),d=Ae(e.inputs,5),p=Ae(e.inputs,6),c=Ae(e.inputs,7);if(i.dims.length===5)throw new Error("Packed QKV is not implemented");if((a==null?void 0:a.dims.length)===5)throw new Error("Packed KV is not implemented");let f=a&&n&&a.dims.length===4&&n.dims.length===4,g=ur(e,r.batchSize,r.numHeads,r.sequenceLength,r.headSize,i,s,0);if(f)return pr(e,g,a,n,u,void 0,p,c,d,r);if(!a||!n)throw new Error("key and value must be provided");let _=ur(e,r.batchSize,r.numHeads,r.kvSequenceLength,r.headSize,a,s,r.hiddenSize),w=ur(e,r.batchSize,r.numHeads,r.kvSequenceLength,r.vHeadSize,n,s,2*r.hiddenSize);pr(e,g,_,w,u,void 0,p,c,d,r)}}),wl,bl,$l,vl,Ta,_h,wh,bh=P(()=>{J(),ie(),ve(),ae(),wl=e=>{if(!e||e.length<1)throw new Error("too few inputs")},bl=(e,t)=>{let r=[],i=t.numOutputs;return e[1].dims[0]>0&&(e[1].getBigInt64Array().forEach(a=>r.push(Number(a))),i=r.length),ce({numOutputs:i,axis:t.axis,splitSizes:r})},$l=e=>`
fn calculateOutputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${e}u; i += 1u ) {
    if (index < ${Z("uniforms.size_in_split_axis","i",e)}) {
        return i;
    }
    }
    return ${e}u;
}`,vl=e=>{let t=e.length,r=[];for(let i=0;i<t;++i){let a=e[i].setByIndices("indices","input[global_idx]");t===1?r.push(a):i===0?r.push(`if (output_number == ${i}u) { ${a} }`):i===t-1?r.push(`else { ${a} }`):r.push(`else if (output_number == ${i}) { ${a} }`)}return`
      fn writeBufferData(output_number: u32, indices: ${e[0].type.indices}, global_idx: u32) {
        ${r.join(`
`)}
      }`},Ta=(e,t)=>{let r=e[0].dims,i=B.size(r),a=e[0].dataType,n=B.normalizeAxis(t.axis,r.length),s=new Array(t.numOutputs),u=M("input",a,r.length),d=new Array(t.numOutputs),p=[],c=[],f=0,g=[{type:12,data:i}];for(let w=0;w<t.numOutputs;w++){f+=t.splitSizes[w],d[w]=f;let b=r.slice();b[n]=t.splitSizes[w],c.push(b),s[w]=j(`output${w}`,a,b.length),p.push({dims:c[w],dataType:e[0].dataType})}g.push({type:12,data:d},...Q(r,...c));let _=w=>`
  ${w.registerUniform("input_size","u32").registerUniform("size_in_split_axis","u32",d.length).declareVariables(u,...s)}
  ${$l(d.length)}
  ${vl(s)}

  ${w.mainStart()}
    ${w.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.input_size")}

    var indices = ${u.offsetToIndices("global_idx")};
    var index = ${u.indicesGet("indices",n)};
    let output_number = calculateOutputIndex(index);
    if (output_number != 0) {
      index -= ${Z("uniforms.size_in_split_axis","output_number - 1u",d.length)};
      ${u.indicesSet("indices",n,"index")};
    }
    writeBufferData(output_number, indices, global_idx);
  }`;return{name:"Split",shaderCache:{hint:t.cacheKey,inputDependencies:["rank"]},getShaderSource:_,getRunData:()=>({outputs:p,dispatchGroup:{x:Math.ceil(i/64)},programUniforms:g})}},_h=(e,t)=>{wl(e.inputs);let r=e.inputs.length===1?t:bl(e.inputs,t);e.compute(Ta(e.inputs,r),{inputs:[0]})},wh=e=>{let t=e.axis,r=e.splitSizes,i=e.numOutputs<0?r.length:e.numOutputs;if(i!==r.length)throw new Error("numOutputs and splitSizes length must be equal");return ce({axis:t,numOutputs:i,splitSizes:r})}}),xl,Gr,$h,vh=P(()=>{J(),ie(),ve(),ae(),xl=(e,t)=>{let[r,i,a,n]=e,{numHeads:s,rotaryEmbeddingDim:u}=t;if(r.dims.length!==3&&r.dims.length!==4)throw new Error(`Input 'x' is expected to have 3 or 4 dimensions, got ${r.dims.length}`);if(!B.areEqual(i.dims,[])&&!B.areEqual(i.dims,[1])&&i.dims.length!==2)throw new Error(`Input 'position_ids' is expected to have 0, 1, or 2 dimensions, got ${i.dims.length}`);if(a.dims.length!==2)throw new Error(`Input 'cos_cache' is expected to have 2 dimensions, got ${a.dims.length}`);if(n.dims.length!==2)throw new Error(`Input 'sin_cache' is expected to have 2 dimensions, got ${n.dims.length}`);if(!B.areEqual(a.dims,n.dims))throw new Error("Inputs 'cos_cache' and 'sin_cache' are expected to have the same shape");if(u>0&&s===0)throw new Error("num_heads must be provided if rotary_embedding_dim is specified");let d=r.dims[0],p=r.dims[r.dims.length-2],c=a.dims[0],f=B.sizeFromDimension(r.dims,1)/p,g=u===0?a.dims[1]*2:f/s;if(u>g)throw new Error("rotary_embedding_dim must be less than or equal to head_size");if(i.dims.length===2){if(d!==i.dims[0])throw new Error(`Input 'position_ids' dimension 0 should be of size batch_size, got ${i.dims[0]}`);if(p!==i.dims[1])throw new Error(`Input 'position_ids' dimension 1 should be of size sequence_length, got ${i.dims[1]}`)}if(g/2!==a.dims[1]&&u/2!==a.dims[1])throw new Error(`Input 'cos_cache' dimension 1 should be same as head_size / 2 or rotary_embedding_dim / 2, got ${a.dims[1]}`);if(p>c)throw new Error("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported")},Gr=(e,t)=>{let{interleaved:r,numHeads:i,rotaryEmbeddingDim:a,scale:n}=t,s=e[0].dims[0],u=B.sizeFromDimension(e[0].dims,1),d=e[0].dims[e[0].dims.length-2],p=u/d,c=e[2].dims[1],f=a===0?c*2:p/i,g=new Array(s,d,p/f,f-c),_=B.computeStrides(g),w=[{type:1,data:n},{type:12,data:g},{type:12,data:_},...e[0].dims.length===3?new Array({type:12,data:[u,p,f,1]}):[],...e[0].dims.length===4?new Array({type:12,data:[u,f,d*f,1]}):[],...Q(e[0].dims,e[1].dims,e[2].dims,e[3].dims,e[0].dims)],b=S=>{let v=M("input",e[0].dataType,e[0].dims.length),$=M("position_ids",e[1].dataType,e[1].dims.length),I=M("cos_cache",e[2].dataType,e[2].dims.length),T=M("sin_cache",e[3].dataType,e[3].dims.length),E=j("output",e[0].dataType,e[0].dims.length);return S.registerUniforms([{name:"scale",type:"f32"},{name:"global_shape",type:"u32",length:g.length},{name:"global_strides",type:"u32",length:_.length},{name:"input_output_strides",type:"u32",length:_.length}]),`
        ${S.declareVariables(v,$,I,T,E)}

        ${S.mainStart(Lt)}
          let half_rotary_emb_dim = uniforms.${I.name}_shape[1];
          let bsnh = global_idx / uniforms.global_strides % uniforms.global_shape;
          let size = uniforms.global_shape[0] * uniforms.global_strides[0];
          ${S.guardAgainstOutOfBoundsWorkgroupSizes("size")}

          if (bsnh[3] < half_rotary_emb_dim) {
            let position_ids_idx =
                ${$.broadcastedIndicesToOffset("bsnh.xy",j("",$.type.tensor,2))};
            let position_id =
                u32(${$.getByOffset("position_ids_idx")}) + select(0, bsnh[1], position_ids_idx == 0);
            let i = dot(bsnh, uniforms.input_output_strides) + select(0, bsnh[3], ${r});
            let j = i + select(half_rotary_emb_dim, 1, ${r});
            let re = ${v.getByOffset("i")} * ${I.get("position_id","bsnh[3]")} -
                ${v.getByOffset("j")} * ${T.get("position_id","bsnh[3]")};
            ${E.setByOffset("i","re")}
            let im = ${v.getByOffset("i")} * ${T.get("position_id","bsnh[3]")} +
                ${v.getByOffset("j")} * ${I.get("position_id","bsnh[3]")};
            ${E.setByOffset("j","im")}
          } else {
            let k = dot(bsnh, uniforms.input_output_strides) + half_rotary_emb_dim;
            ${E.setByOffset("k",v.getByOffset("k"))}
          }
        }`};return{name:"RotaryEmbedding",shaderCache:{hint:ce({interleaved:r}).cacheKey,inputDependencies:["rank","rank","rank","rank"]},getShaderSource:b,getRunData:()=>({outputs:[{dims:e[0].dims,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(B.size(g)/Lt)},programUniforms:w})}},$h=(e,t)=>{xl(e.inputs,t),e.compute(Gr(e.inputs,t))}}),Sl,kl,Zi,Tl,xh,S0=P(()=>{ve(),J(),Va(),yh(),bh(),yt(),vh(),ae(),Sl=(e,t)=>{if(t.doRotary&&e.length<=7)throw new Error("cos_cache and sin_cache inputs are required if do_rotary is specified");let r=e[0],i=e[1],a=e[2],n=e[3],s=e[4];if(t.doRotary!==0&&e.length<=7)throw new Error("cos_cast and sin_cache are expected if do_rotary attribute is non-zero");if(t.localWindowSize!==-1)throw new Error("Local attention is not supported");if(t.softcap!==0)throw new Error("Softcap is not supported");if(t.rotaryInterleaved!==0)throw new Error("Rotary interleaved is not supported");if(t.smoothSoftmax)throw new Error("Smooth softmax is not supported");if(r.dims.length!==3&&r.dims.length!==5)throw new Error("Input query is expected to have 3 or 5 dimensions");let u=!1,d=r.dims[0],p=r.dims[1],c=r.dims.length===3?u?r.dims[2]/3:r.dims[2]:t.numHeads*r.dims[4],f=p,g=0,_=!i||i.dims.length===0,w=Math.floor(_?c/(t.numHeads+2*t.kvNumHeads):c/t.numHeads);_&&(c=w*t.numHeads);let b=n&&n.dims.length!==0,S=s&&s.dims.length!==0;if(b&&n.dims.length===4&&n.dims[0]===d&&n.dims[1]!==t.kvNumHeads&&n.dims[2]===t.kvNumHeads&&n.dims[3]===w)throw new Error("BSNH pastKey/pastValue is not supported");if(b&&S){if(n.dims.length!==4)throw new Error('Input "past_key" is expected to have 4 dimensions');if(s.dims.length!==4)throw new Error('Input "past_value" is expected to have 4 dimensions');g=n.dims[2]}else if(b||S)throw new Error('Input "past_key" and "past_value" shall be both present or both absent');let v=1;if(i&&i.dims.length>0){if(r.dims.length!==3)throw new Error('Input "query" is expected to have 3 dimensions when key is given');if(i.dims.length<3||i.dims.length>5)throw new Error('Input "key" is expected to have 3, 4, or 5 dimensions');if(r.dims[0]!==i.dims[0])throw new Error('Input "query" and "key" shall have same dim 0 (batch size)');if(i.dims.length===3){if(r.dims[2]%i.dims[2]!==0)throw new Error('Dimension 2 of "query" should be a multiple of "key"');f=i.dims[1]}else if(i.dims.length===5){if(i.dims[2]!==t.numHeads||i.dims[3]!==2||i.dims[4]!==w)throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');if(a)throw new Error('Expect "value" be none when "key" has packed kv format.');f=i.dims[1]}else{if(i.dims[1]!==t.numHeads||i.dims[3]!==w)throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');f=i.dims[2]}}else{if(r.dims.length!==3&&r.dims.length!==5)throw new Error('Input "query" is expected to have 3 or 5 dimensions when key is empty');if(r.dims.length===5&&(r.dims[2]!==t.numHeads||r.dims[3]!==3))throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');v=3}let $=0,I=!1,T=t.kvNumHeads?w*t.kvNumHeads:c;if(a&&a.dims.length>0){if(a.dims.length!==3&&a.dims.length!==4)throw new Error('Input "value" is expected to have 3 or 4 dimensions');if(r.dims[0]!==a.dims[0])throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');if(a.dims.length===3){if(f!==a.dims[1])throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');T=a.dims[2]}else{if(f!==a.dims[2])throw new Error('Input "past_key" and "past_value" shall have the same dim 2 (kv_sequence_length)');T=a.dims[1]*a.dims[3],I=!0}}let E=e.length>4?e[5]:void 0;if(E&&E.dims.length!==1&&E.dims[0]!==d)throw new Error('Input "seqlens" is expected to have 1 dimension and the same dim 0 as batch_size');return{batchSize:d,sequenceLength:p,pastSequenceLength:g,kvSequenceLength:f,totalSequenceLength:-1,maxSequenceLength:-1,inputHiddenSize:0,hiddenSize:c,vHiddenSize:T,headSize:w,vHeadSize:Math.floor(T/t.kvNumHeads),numHeads:t.numHeads,kvNumHeads:t.kvNumHeads,nReps:t.numHeads/t.kvNumHeads,pastPresentShareBuffer:!1,maskType:$,scale:t.scale,broadcastResPosBias:!1,passPastInKv:I,qkvFormat:v}},kl=ce({perm:[0,2,1,3]}),Zi=(e,t,r)=>{let i=t,a=r.kvNumHeads;return t.dims.length===3&&r.kvSequenceLength!==0&&(i=t.reshape([r.batchSize,r.kvSequenceLength,a,r.headSize]),i=e.compute(De(i,kl.perm),{inputs:[i],outputs:[-1]})[0]),i},Tl=(e,t,r,i)=>{let a=7,n=["type","type"],s=[e*t],u=e*t,d=[{type:12,data:u},{type:12,data:t},{type:12,data:e}],p=c=>{let f=M("seq_lens",r.dataType,r.dims),g=M("total_seq_lens",i.dataType,i.dims),_=j("pos_ids",a,s),w=[{name:"output_size",type:"u32"},{name:"sequence_length",type:"u32"},{name:"batch_size",type:"u32"}];return`
  ${c.registerUniforms(w).declareVariables(f,g,_)}
  ${c.mainStart()}
    ${c.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
    let total_sequence_length = u32(${g.getByOffset("0")});
    let is_subsequent_prompt = uniforms.sequence_length > 1 && uniforms.sequence_length != total_sequence_length;
    let is_first_prompt = !is_subsequent_prompt && uniforms.sequence_length == total_sequence_length;
    let batch_idx = global_idx / uniforms.sequence_length;
    let sequence_idx = i32(global_idx % uniforms.sequence_length);
    var pos_id: i32 = 0;
    let seqlen = ${f.getByOffset("batch_idx")};
    let total_seqlen = seqlen + 1;
    if (is_first_prompt) {
      if (sequence_idx < total_seqlen) {
        pos_id = sequence_idx;
      } else {
        pos_id = 1;
      }
      ${_.setByOffset("global_idx","pos_id")}
    } else if (is_subsequent_prompt) {
      let past_seqlen = total_seqlen - i32(uniforms.sequence_length);
      if (past_seqlen + sequence_idx < total_seqlen) {
        pos_id = past_seqlen + sequence_idx;
      } else {
        pos_id = 1;
      }
      ${_.setByOffset("global_idx","pos_id")}
    } else if (global_idx < uniforms.batch_size) {
      ${_.setByOffset("global_idx","seqlen")}
    };
  }
  `};return{name:"GeneratePositionIds",shaderCache:{hint:`${e};${t}`,inputDependencies:n},getRunData:()=>({outputs:[{dims:s,dataType:a}],dispatchGroup:{x:Math.ceil(u/64)},programUniforms:d}),getShaderSource:p}},xh=(e,t)=>{var T;let r=Sl(e.inputs,t);if(e.inputs[0].dims.length===5)throw new Error("Packed QKV is not implemented");if(((T=e.inputs[1])==null?void 0:T.dims.length)===5)throw new Error("Packed KV is not implemented");let i=e.inputs[0],a=e.inputs[1]&&e.inputs[1].dims.length>0?e.inputs[1]:void 0,n=e.inputs[2]&&e.inputs[2].dims.length>0?e.inputs[2]:void 0,s=e.inputs[3]&&e.inputs[3].dims.length!==0?e.inputs[3]:void 0,u=e.inputs[4]&&e.inputs[4].dims.length!==0?e.inputs[4]:void 0,d=e.inputs.length>4?e.inputs[5]:void 0,p=e.inputs.length>5?e.inputs[6]:void 0,c=r.kvNumHeads?r.kvNumHeads:r.numHeads,f=ce({axis:2,numOutputs:3,splitSizes:[r.numHeads*r.headSize,c*r.headSize,c*r.headSize]}),[g,_,w]=!a&&!n?e.compute(Ta([i],f),{inputs:[i],outputs:[-1,-1,-1]}):[i,a,n],b,S;if(t.doRotary){let E=e.compute(Tl(r.batchSize,r.sequenceLength,d,p),{inputs:[d,p],outputs:[-1]})[0],A=e.inputs[7],C=e.inputs[8],O=ce({interleaved:t.rotaryInterleaved!==0,numHeads:r.numHeads,rotaryEmbeddingDim:0,scale:t.scale}),U=[g,E,A,C],x=[-1];b=e.compute(Gr(U,O),{inputs:U,outputs:x})[0],U.splice(0,1,_);let Y=ce({interleaved:t.rotaryInterleaved!==0,numHeads:r.kvNumHeads,rotaryEmbeddingDim:0,scale:t.scale});S=e.compute(Gr(U,Y),{inputs:U,outputs:x})[0]}let v=ur(e,r.batchSize,r.numHeads,r.sequenceLength,r.headSize,t.doRotary?b:g,void 0,0),$=Zi(e,t.doRotary?S:_,r),I=Zi(e,w,r);pr(e,v,$,I,void 0,void 0,s,u,void 0,r,d,p)}}),Qi,Il,El,Sh,k0=P(()=>{J(),ie(),yt(),ae(),Qi=(e,t,r,i,a,n,s,u)=>{let d=$e(n),p=d===1?"f32":`vec${d}f`,c=d===1?"vec2f":`mat2x${d}f`,f=a*s,g=64;f===1&&(g=256);let _=[a,s,n/d],w=[a,s,2],b=["rank","type","type"],S=[];S.push(...Q(_,w));let v=$=>{let I=M("x",t.dataType,3,d),T=M("scale",r.dataType,r.dims),E=M("bias",i.dataType,i.dims),A=j("output",1,3,2),C=[I,T,E,A];return`
  var<workgroup> workgroup_shared : array<${c}, ${g}>;
  const workgroup_size = ${g}u;
  ${$.declareVariables(...C)}
  ${$.mainStart(g)}
    let batch = workgroup_index / uniforms.x_shape[1];
    let channel = workgroup_index % uniforms.x_shape[1];
    let hight = uniforms.x_shape[2];
    // initialize workgroup memory
    var sum = ${p}(0);
    var squared_sum = ${p}(0);
    for (var h = local_idx; h < hight; h += workgroup_size) {
      let value = ${p}(${I.get("batch","channel","h")});
      sum += value;
      squared_sum += value * value;
    }
    workgroup_shared[local_idx] = ${c}(sum, squared_sum);
    workgroupBarrier();

    for (var currSize = workgroup_size >> 1;  currSize > 0; currSize = currSize >> 1) {
      if (local_idx < currSize) {
        workgroup_shared[local_idx] = workgroup_shared[local_idx] + workgroup_shared[local_idx + currSize];
      }
      workgroupBarrier();
    }
    if (local_idx == 0) {
      let sum_final = ${gt("workgroup_shared[0][0]",d)} / f32(hight * ${d});
      let squared_sum_final = ${gt("workgroup_shared[0][1]",d)} / f32(hight * ${d});

      let inv_std_dev = inverseSqrt(squared_sum_final - sum_final * sum_final + f32(${u}));
      let channel_scale = inv_std_dev * f32(scale[channel]);
      let channel_shift = f32(bias[channel]) - sum_final * channel_scale;
      output[workgroup_index] = vec2f(channel_scale, channel_shift);
    }
  }`};return e.compute({name:"InstanceNormComputeChannelScaleShift",shaderCache:{hint:`${d};${u};${g}`,inputDependencies:b},getRunData:()=>({outputs:[{dims:w,dataType:1}],dispatchGroup:{x:f},programUniforms:S}),getShaderSource:v},{inputs:[t,r,i],outputs:[-1]})[0]},Il=(e,t,r)=>{let i=t[0].dims,a=i,n=2,s=i[0],u=i[1],d=B.sizeFromDimension(i,n),p=$e(d),c=B.size(a)/p,f=Qi(e,t[0],t[1],t[2],s,d,u,r.epsilon),g=[s,u,d/p],_=[s,u],w=["type","none"],b=S=>{let v=M("x",t[0].dataType,g.length,p),$=M("scale_shift",1,_.length,2),I=j("output",t[0].dataType,g.length,p),T=[v,$,I];return`
  ${S.registerUniform("output_size","u32").declareVariables(...T)}
  ${S.mainStart()}
  ${S.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
      let outputIndices = ${I.offsetToIndices("global_idx")};
      let batch = outputIndices[0];
      let channel = outputIndices[1];
      let scale_shift = ${$.getByIndices("vec2<u32>(batch, channel)")};
      let value = ${v.getByOffset("global_idx")} * ${I.type.value}(scale_shift.x) + ${I.type.value}(scale_shift.y);
      ${I.setByOffset("global_idx","value")};
  }`};e.compute({name:"InstanceNormalization",shaderCache:{hint:`${p}`,inputDependencies:w},getRunData:()=>({outputs:[{dims:a,dataType:t[0].dataType}],dispatchGroup:{x:Math.ceil(c/64)},programUniforms:[{type:12,data:c},...Q(g,_,g)]}),getShaderSource:b},{inputs:[t[0],f]})},El=(e,t,r)=>{let i=t[0].dims,a=i,n=i[0],s=i[i.length-1],u=B.sizeFromDimension(i,1)/s,d=$e(s),p=B.size(a)/d,c=[{type:12,data:u},{type:12,data:Math.floor(s/d)}],f=["type","type"],g=!1,_=[0,i.length-1];for(let v=0;v<i.length-2;v++)g=g||i[v+1]!==1,_.push(v+1);g=g&&i[i.length-1]!==1;let w=g?e.compute(De(e.inputs[0],_),{inputs:[e.inputs[0]],outputs:[-1]})[0]:e.inputs[0].reshape(Array.from({length:i.length},(v,$)=>i[_[$]])),b=Qi(e,w,t[1],t[2],n,u,s,r.epsilon),S=v=>{let $=ke(t[0].dataType),I=d===1?"vec2f":`mat${d}x2f`,T=C=>{let O=C===0?"x":"y",U=d===1?"f32":`vec${d}f`;switch(d){case 1:return`${$}(${U}(scale.${O}))`;case 2:return`vec2<${$}>(${U}(scale[0].${O}, scale[1].${O}))`;case 4:return`vec4<${$}>(${U}(scale[0].${O}, scale[1].${O}, scale[2].${O}, scale[3].${O}))`;default:throw new Error(`Not supported compoents ${d}`)}},E=M("input",t[0].dataType,t[0].dims,d),A=j("output",t[0].dataType,a,d);return`
  @group(0) @binding(0) var<storage, read> input : array<${E.type.storage}>;
  @group(0) @binding(1) var<storage, read> scale_input : array<${I}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${A.type.storage}>;
  struct Uniforms {H: u32, C : u32};
  @group(0) @binding(3) var<uniform> uniforms: Uniforms;

  ${v.mainStart()}
    let current_image_number = global_idx / (uniforms.C * uniforms.H);
    let current_channel_number = global_idx % uniforms.C;

    let scale_offset = current_image_number * uniforms.C + current_channel_number;
    let scale = scale_input[scale_offset];
    output[global_idx] = fma(input[global_idx], ${T(0)}, ${T(1)});
  }`};e.compute({name:"InstanceNormalizationNHWC",shaderCache:{hint:`${d}`,inputDependencies:f},getRunData:()=>({outputs:[{dims:a,dataType:t[0].dataType}],dispatchGroup:{x:Math.ceil(p/64)},programUniforms:c}),getShaderSource:S},{inputs:[t[0],b]})},Sh=(e,t)=>{t.format==="NHWC"?El(e,e.inputs,t):Il(e,e.inputs,t)}}),zl,Cl,kh,T0=P(()=>{J(),ie(),ae(),zl=e=>{if(!e||e.length<2)throw new Error("layerNorm requires at least 2 inputs.")},Cl=(e,t,r)=>{let i=t.simplified,a=e[0].dims,n=e[1],s=!i&&e[2],u=a,d=B.normalizeAxis(t.axis,a.length),p=B.sizeToDimension(a,d),c=B.sizeFromDimension(a,d),f=B.size(n.dims),g=s?B.size(s.dims):0;if(f!==c||s&&g!==c)throw new Error(`Size of X.shape()[axis:] == ${c}.
       Size of scale and bias (if provided) must match this.
       Got scale size of ${f} and bias size of ${g}`);let _=[];for(let E=0;E<a.length;++E)E<d?_.push(a[E]):_.push(1);let w=$e(c),b=["type","type"],S=[{type:12,data:p},{type:1,data:c},{type:12,data:Math.floor(c/w)},{type:1,data:t.epsilon}];s&&b.push("type");let v=r>1,$=r>2,I=E=>{let A=ke(e[0].dataType),C=[M("x",e[0].dataType,e[0].dims,w),M("scale",n.dataType,n.dims,w)];s&&C.push(M("bias",s.dataType,s.dims,w)),C.push(j("output",e[0].dataType,u,w)),v&&C.push(j("mean_data_output",1,_)),$&&C.push(j("inv_std_output",1,_));let O=[{name:"norm_count",type:"u32"},{name:"norm_size",type:"f32"},{name:"norm_size_vectorized",type:"u32"},{name:"epsilon",type:"f32"}];return`
  ${E.registerUniforms(O).declareVariables(...C)}
  ${E.mainStart()}
    ${E.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.norm_count")}
    let offset = global_idx * uniforms.norm_size_vectorized;
    var mean_vector = ${ya("f32",w)};
    var mean_square_vector = ${ya("f32",w)};

    for (var h: u32 = 0u; h < uniforms.norm_size_vectorized; h++) {
      let value = ${qt(A,w,"x[h + offset]")};
      mean_vector += value;
      mean_square_vector += value * value;
    }
    let mean = ${gt("mean_vector",w)} / uniforms.norm_size;
    let inv_std_dev = inverseSqrt(${gt("mean_square_vector",w)} / uniforms.norm_size ${i?"":"- mean * mean"} + uniforms.epsilon);

    for (var j: u32 = 0; j < uniforms.norm_size_vectorized; j++) {
      let f32input = ${qt(A,w,"x[j + offset]")};
      let f32scale = ${qt(A,w,"scale[j]")};
      output[j + offset] = ${C[0].type.value}((f32input ${i?"":"- mean"}) * inv_std_dev * f32scale
        ${s?`+ ${qt(A,w,"bias[j]")}`:""}
      );
    }

    ${v?"mean_data_output[global_idx] = mean":""};
    ${$?"inv_std_output[global_idx] = inv_std_dev":""};
  }`},T=[{dims:u,dataType:e[0].dataType}];return v&&T.push({dims:_,dataType:1}),$&&T.push({dims:_,dataType:1}),{name:"LayerNormalization",shaderCache:{hint:`${w};${r};${i}`,inputDependencies:b},getRunData:()=>({outputs:T,dispatchGroup:{x:Math.ceil(p/64)},programUniforms:S}),getShaderSource:I}},kh=(e,t)=>{zl(e.inputs),e.compute(Cl(e.inputs,t,e.outputCount))}}),Al,Th,I0=P(()=>{ie(),Ka(),Za(),Al=e=>{if(!e||e.length!==2)throw new Error("MatMul requires 2 inputs.");if(e[0].dims[e[0].dims.length-1]!==e[1].dims[e[1].dims.length-2])throw new Error("shared dimension does not match.")},Th=e=>{Al(e.inputs);let t=Wt.calcShape(e.inputs[0].dims,e.inputs[1].dims,!0);if(!t)throw new Error("Can't use matmul on the given tensors");let r=t[t.length-1],i=e.inputs[0].dims[e.inputs[0].dims.length-1];if(r<8&&i<8)e.compute(ja(e.inputs,{activation:""},t));else{let a=t[t.length-2],n=B.size(e.inputs[0].dims.slice(0,-2)),s=B.size(e.inputs[1].dims.slice(0,-2));if(n!==1&&a===1&&s===1){let u=e.inputs[0].reshape([1,n,i]),d=e.inputs[1].reshape([1,i,r]),p=[1,n,r],c=[u,d];e.compute(Vr(c,{activation:""},t,p),{inputs:c})}else e.compute(Vr(e.inputs,{activation:""},t))}}}),Ol,Rl,Bl,Ih,Eh,E0=P(()=>{J(),ie(),ve(),ae(),Ol=(e,t)=>{if(e.length<3||e.length>4)throw new Error("MatMulNBits requires 3 or 4 inputs");let r=e[0],i=r.dims.length;if(r.dims[i-1]!==t.k)throw new Error("The last dim of input shape does not match the k value");let a=Math.floor((t.k+t.blockSize-1)/t.blockSize),n=t.blockSize/8*t.bits,s=e[1];if(!B.areEqual(s.dims,[t.n,a,n]))throw new Error("The second inputs must be 3D tensor with shape N X nBlocksPerCol X blobSize");let u=e[2].dims;if(B.size(u)!==t.n*a)throw new Error("scales input size error.");if(e.length===4){let d=e[3].dims,p=t.n*(t.bits===8?a:Math.floor((a*t.bits+7)/8));if(B.size(d)!==p)throw new Error("zeroPoints input size error.")}},Rl=(e,t)=>{let r=e[0].dims,i=r.length,a=r[i-2],n=t.k,s=t.n,u=r.slice(0,i-2),d=B.size(u),p=e[1].dims[2]/4,c=e[0].dataType,f=$e(t.k),g=$e(p),_=$e(s),w=u.concat([a,s]),b=a>1&&s/_%2===0?2:1,S=B.size(w)/_/b,v=64,$=[],I=[d,a,n/f],T=B.convertShape(e[1].dims).slice();T.splice(-1,1,p/g),$.push(...Q(I)),$.push(...Q(T)),$.push(...Q(e[2].dims)),e.length===4&&$.push(...Q(B.convertShape(e[3].dims)));let E=[d,a,s/_];$.push(...Q(E));let A=C=>{let O=I.length,U=M("a",e[0].dataType,O,f),x=M("b",12,T.length,g),Y=M("scales",e[2].dataType,e[2].dims.length),G=[U,x,Y],V=e.length===4?M("zero_points",12,e[3].dims.length):void 0;V&&G.push(V);let te=E.length,ee=j("output",e[0].dataType,te,_),F=ke(e[0].dataType),R=(()=>{switch(f){case 1:return`array<${F}, 8>`;case 2:return`mat4x2<${F}>`;case 4:return`mat2x4<${F}>`;default:throw new Error(`${f}-component is not supported.`)}})(),q=()=>{let D=`
          // reuse a data
            var input_offset = ${U.indicesToOffset(`${U.type.indices}(batch, row, word_offset)`)};
            var a_data: ${R};
            for (var j: u32 = 0; j < ${8/f}; j++) {
              a_data[j] = ${U.getByOffset("input_offset")};
              input_offset++;
            }
          `;for(let L=0;L<_*b;L++)D+=`
            b_value = ${g===1?`b${L}_data`:`b${L}_data[i]`};
            b_value_lower = unpack4xU8(b_value & b_mask);
            b_value_upper = unpack4xU8((b_value >> 4) & b_mask);
            b_quantized_values = ${R}(${Array.from({length:4},(K,re)=>`${F}(b_value_lower[${re}]), ${F}(b_value_upper[${re}])`).join(", ")});
            b_dequantized_values = ${f===1?`${R}(${Array.from({length:8},(K,re)=>`(b_quantized_values[${re}] - ${V?`zero_point${L}`:"zero_point"}) * scale${L}`).join(", ")});`:`(b_quantized_values - ${R}(${Array(8).fill(`${V?`zero_point${L}`:"zero_point"}`).join(",")})) * scale${L};`};
            workgroup_shared[local_id.x * ${b} + ${Math.floor(L/_)}]${_>1?`[${L%_}]`:""} += ${Array.from({length:8/f},(K,re)=>`${f===1?`a_data[${re}] * b_dequantized_values[${re}]`:`dot(a_data[${re}], b_dequantized_values[${re}])`}`).join(" + ")};
          `;return D},X=()=>{let D=`
            var col_index = col * ${_};
            ${V?`
            let zero_point_bytes_per_col = (nBlocksPerCol + 1) / 2;
            var zero_point_byte_count: u32;
            var zero_point_word_index: u32;
            var zero_point_byte_offset: u32;
            let zero_point_nibble_offset: u32 = block & 0x1u;
            var zero_point_bits_offset: u32;
            var zero_point_word: u32;`:`
            // The default zero point is 8 for unsigned 4-bit quantization.
            let zero_point = ${F}(8);`}
            `;for(let L=0;L<_*b;L++)D+=`
            let scale${L} = ${Y.getByOffset("col_index * nBlocksPerCol + block")};
            ${V?`
            zero_point_byte_count = col_index * zero_point_bytes_per_col + (block >> 0x1u);
            zero_point_word_index = zero_point_byte_count >> 0x2u;
            zero_point_byte_offset = zero_point_byte_count & 0x3u;
            zero_point_bits_offset = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);
            zero_point_word = ${V.getByOffset("zero_point_word_index")} >> zero_point_bits_offset;
            let zero_point${L} = ${F}((zero_point_word) & 0xFu);`:""}
            col_index += 1;`;return D},_e=()=>{let D=`col_index = col * ${_};`;for(let L=0;L<_*b;L++)D+=`
            let b${L}_data = ${x.getByIndices(`${x.type.indices}(col_index, block, word)`)};
            col_index += 1;`;return D+=`
            var b_value: u32;
            let b_mask: u32 = 0x0F0F0F0Fu;
            var b_value_lower: vec4<u32>;
            var b_value_upper: vec4<u32>;
            var b_quantized_values: ${R};
            var b_dequantized_values: ${R};`,D};return`
        var<workgroup> workgroup_shared: array<${ee.type.value}, ${b*v}>;
        ${C.declareVariables(...G,ee)}
        ${C.mainStart([v,1,1])}
          let output_indices = ${ee.offsetToIndices(`(global_idx / ${v}) * ${b}`)};
          let col = output_indices[2];
          let row = output_indices[1];
          let batch = output_indices[0];
          let nBlocksPerCol = uniforms.b_shape[1];

          for (var block = local_id.x; block < nBlocksPerCol; block += ${v}) {
            //process one block
            var word_offset: u32 = block * ${t.blockSize/f};
            ${X()}
            for (var word: u32 = 0; word < ${p}; word += ${g}) {
              ${_e()}
              for (var i: u32 = 0; i < ${g}; i++) {
                ${q()}
                word_offset += ${8/f};
              }
            }
          }
          workgroupBarrier();

          if (local_id.x < ${b}) {
            var output_value: ${ee.type.value} = ${ee.type.value}(0);
            var workgroup_shared_offset: u32 = local_id.x;
            for (var b: u32 = 0u; b < ${v}u; b++) {
              output_value += workgroup_shared[workgroup_shared_offset];
              workgroup_shared_offset += ${b};
            }
            ${ee.setByIndices(`${ee.type.indices}(batch, row, col + local_id.x)`,"output_value")};
          }
        }`};return{name:"MatMulNBits",shaderCache:{hint:`${t.blockSize};${t.bits};${f};${g};${_};${b};${v}`,inputDependencies:Array(e.length).fill("rank")},getRunData:()=>({outputs:[{dims:w,dataType:c}],dispatchGroup:{x:S},programUniforms:$}),getShaderSource:A}},Bl=(e,t)=>{let r=e[0].dims,i=r.length,a=r[i-2],n=t.k,s=t.n,u=r.slice(0,i-2),d=B.size(u),p=e[1].dims[2]/4,c=e[0].dataType,f=$e(t.k),g=$e(p),_=u.concat([a,s]),w=128,b=s%8===0?8:s%4===0?4:1,S=w/b,v=S*g*8,$=v/f,I=v/t.blockSize,T=B.size(_)/b,E=[],A=[d,a,n/f],C=B.convertShape(e[1].dims).slice();C.splice(-1,1,p/g),E.push(...Q(A)),E.push(...Q(C)),E.push(...Q(e[2].dims)),e.length===4&&E.push(...Q(B.convertShape(e[3].dims)));let O=[d,a,s];E.push(...Q(O));let U=x=>{let Y=A.length,G=M("a",e[0].dataType,Y,f),V=M("b",12,C.length,g),te=M("scales",e[2].dataType,e[2].dims.length),ee=[G,V,te],F=e.length===4?M("zero_points",12,e[3].dims.length):void 0;F&&ee.push(F);let R=O.length,q=j("output",e[0].dataType,R),X=ke(e[0].dataType),_e=()=>{switch(f){case 1:return`
          let a_data0 = vec4<${X}>(sub_a[word_offset], sub_a[word_offset + 1], sub_a[word_offset + 2], sub_a[word_offset + 3]);
          let a_data1 = vec4<${X}>(sub_a[word_offset + 4], sub_a[word_offset + 5], sub_a[word_offset + 6], sub_a[word_offset + 7]);`;case 2:return`
          let a_data0 = vec4<${X}>(sub_a[word_offset], sub_a[word_offset + 1]);
          let a_data1 = vec4<${X}>(sub_a[word_offset + 2], sub_a[word_offset + 3]);`;case 4:return`
          let a_data0 = sub_a[word_offset];
          let a_data1 = sub_a[word_offset + 1];`;default:throw new Error(`${f}-component is not supported.`)}};return`
        var<workgroup> sub_a: array<${G.type.value}, ${$}>;
        var<workgroup> inter_results: array<array<${q.type.value}, ${S}>, ${b}>;
        ${x.declareVariables(...ee,q)}
        ${x.mainStart([S,b,1])}
          let output_indices = ${q.offsetToIndices(`workgroup_index * ${b}`)};
          let col = output_indices[2];
          let row = output_indices[1];
          let batch = output_indices[0];
          let n_blocks_per_col = uniforms.b_shape[1];
          let num_tiles =  (n_blocks_per_col - 1) / ${I} + 1;

          // Loop over shared dimension.
          for (var tile: u32 = 0; tile < num_tiles; tile += 1) {
            let a_col_start = tile * ${$};
            // load one tile A data into shared memory.
            for (var a_offset = local_idx; a_offset < ${$}; a_offset += ${w})
            {
              let a_col = a_col_start + a_offset;
              if (a_col < uniforms.a_shape[2])
              {
                sub_a[a_offset] = ${G.getByIndices(`${G.type.indices}(batch, row, a_col)`)};
              } else {
                sub_a[a_offset] = ${G.type.value}(0);
              }
            }
            workgroupBarrier();

            // each thread process one block
            let b_row = col + local_id.y;
            let block = tile * ${I} + local_id.x;
            ${F?`
            let zero_point_bytes_per_col = (n_blocks_per_col + 1) / 2;
            let zero_point_byte_count = b_row * zero_point_bytes_per_col + (block >> 0x1u);
            let zero_point_word_index = zero_point_byte_count >> 0x2u;
            let zero_point_byte_offset = zero_point_byte_count & 0x3u;
            let zero_point_nibble_offset: u32 = block & 0x1u;
            let zero_point_bits_offset = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);
            let zero_point_word = ${F.getByOffset("zero_point_word_index")} >> zero_point_bits_offset;
            let zero_point = ${X}((zero_point_word) & 0xFu);`:`
            // The default zero point is 8 for unsigned 4-bit quantization.
            let zero_point = ${X}(8);`}
            let scale = ${te.getByOffset("b_row * n_blocks_per_col + block")};
            let b_data = ${V.getByIndices(`${V.type.indices}(b_row, block, 0)`)};
            var word_offset = local_id.x * ${t.blockSize/f};
            for (var i: u32 = 0; i < ${g}; i++) {
              ${_e()}
              let b_value = ${g===1?"b_data":"b_data[i]"};
              let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);
              let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);
              let b_quantized_values = mat2x4<${X}>(${Array.from({length:4},(D,L)=>`${X}(b_value_lower[${L}]), ${X}(b_value_upper[${L}])`).join(", ")});
              let b_dequantized_values = (b_quantized_values - mat2x4<${X}>(${Array(8).fill("zero_point").join(",")})) * scale;
              inter_results[local_id.y][local_id.x] += ${Array.from({length:2},(D,L)=>`${`dot(a_data${L}, b_dequantized_values[${L}])`}`).join(" + ")};
              word_offset += ${8/f};
            }
            workgroupBarrier();
          }

          if (local_idx < ${b}) {
            var output_value: ${q.type.value} = ${q.type.value}(0);
            for (var b = 0u; b < ${S}; b++) {
              output_value += inter_results[local_idx][b];
            }
            if (col + local_idx < uniforms.output_shape[2])
            {
              ${q.setByIndices(`${q.type.indices}(batch, row, col + local_idx)`,"output_value")}
            }
          }
        }`};return{name:"BlockwiseMatMulNBits32",shaderCache:{hint:`${t.blockSize};${f};${g};${S};${b}`,inputDependencies:Array(e.length).fill("rank")},getRunData:()=>({outputs:[{dims:_,dataType:c}],dispatchGroup:{x:T},programUniforms:E}),getShaderSource:U}},Ih=(e,t)=>{Ol(e.inputs,t),t.blockSize===32&&e.adapterInfo.isVendor("intel")&&e.adapterInfo.isArchitecture("gen-12lp")?e.compute(Bl(e.inputs,t)):e.compute(Rl(e.inputs,t))},Eh=e=>ce(e)}),Nl,Dl,Ml,Ul,Pl,ql,Wl,Ll,zh,z0=P(()=>{J(),ie(),ae(),Nl=e=>{if(!e||e.length<1)throw new Error("Too few inputs");if(e[0].dataType!==1&&e[0].dataType!==10)throw new Error("Input type must be float or float16.");if(e.length>=2){let t=e[0].dims.length*2===e[1].dims[0];if(e.length===4&&(t=e[3].dims[0]*2===e[1].dims[0]),!t)throw new Error("The pads should be a 1D tensor of shape [2 * input_rank] or [2 * num_axes].")}},Dl=(e,t,r)=>{let i="";for(let a=t-1;a>=0;--a)i+=`
            k = i32(${e.indicesGet("indices",a)}) - ${Z("uniforms.pads",a,r)};
            if (k < 0) {
              break;
            }
            if (k >= i32(${Z("uniforms.x_shape",a,t)})) {
              break;
            }
            offset += k * i32(${Z("uniforms.x_strides",a,t)});
        `;return`
          value = ${e.type.value}(uniforms.constant_value);
          for (var i = 0; i < 1; i++) {
            var offset = 0;
            var k = 0;
            ${i}
            value = x[offset];
          }
      `},Ml=(e,t,r)=>{let i="";for(let a=t-1;a>=0;--a)i+=`
                k = i32(${e.indicesGet("indices",a)}) - ${Z("uniforms.pads",a,r)};
                if (k < 0) {
                  k = -k;
                }
                {
                  let _2n_1 = 2 * (i32(${Z("uniforms.x_shape",a,t)}) - 1);
                  k = k % _2n_1;
                  if(k >= i32(${Z("uniforms.x_shape",a,t)})) {
                    k = _2n_1 - k;
                  }
                }
                offset += k * i32(${Z("uniforms.x_strides",a,t)});
            `;return`
              var offset = 0;
              var k = 0;
              ${i}
              value = x[offset];
          `},Ul=(e,t,r)=>{let i="";for(let a=t-1;a>=0;--a)i+=`
                k = i32(${e.indicesGet("indices",a)}) - ${Z("uniforms.pads",a,r)};
                if (k < 0) {
                  k = 0;
                }
                if (k >= i32(${Z("uniforms.x_shape",a,t)})) {
                  k = i32(${Z("uniforms.x_shape",a,t)}) - 1;
                }
                offset += k * i32(${Z("uniforms.x_strides",a,t)});
            `;return`
              var offset = 0;
              var k = 0;
              ${i}
              value = x[offset];
          `},Pl=(e,t,r)=>{let i="";for(let a=t-1;a>=0;--a)i+=`
                k = i32(${e.indicesGet("indices",a)}) - ${Z("uniforms.pads",a,r)};
                if (k < 0)  {
                  k += i32(${Z("uniforms.x_shape",a,t)}]);
                }
                if (k >= i32(${Z("uniforms.x_shape",a,t)})) {
                  k -= i32(${Z("uniforms.x_shape",a,t)});
                }
                offset += k * i32(${Z("uniforms.x_strides",a,t)});
            `;return`
              var offset = 0;
              var k = 0;
              ${i}
              value = x[offset];
          `},ql=(e,t,r)=>{switch(r.mode){case 0:return Dl(e,t,r.pads.length);case 1:return Ml(e,t,r.pads.length);case 2:return Ul(e,t,r.pads.length);case 3:return Pl(e,t,r.pads.length);default:throw new Error("Invalid mode")}},Wl=(e,t)=>{let r=B.padShape(e[0].dims.slice(),t.pads),i=e[0].dims,a=B.size(r),n=[{type:12,data:a},{type:6,data:t.pads}],s=e.length>=3&&e[2].data;t.mode===0&&n.push({type:s?e[2].dataType:1,data:t.value}),n.push(...Q(e[0].dims,r));let u=["rank"],d=p=>{let c=j("output",e[0].dataType,r.length),f=M("x",e[0].dataType,i.length),g=f.type.value,_=ql(c,i.length,t),w=[{name:"output_size",type:"u32"},{name:"pads",type:"i32",length:t.pads.length}];return t.mode===0&&w.push({name:"constant_value",type:s?g:"f32"}),`
            ${p.registerUniforms(w).declareVariables(f,c)}
            ${p.mainStart()}
            ${p.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}

            let indices = ${c.offsetToIndices("global_idx")};

            var value = ${g}(0);
            ${_}
            output[global_idx] = value;
        }`};return{name:"Pad",shaderCache:{hint:`${t.mode}${s}`,inputDependencies:u},getRunData:()=>({outputs:[{dims:r,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(B.size(r)/64)},programUniforms:n}),getShaderSource:d}},Ll=(e,t)=>{if(e.length>1){let r=e[1].getBigInt64Array(),i=e.length>=3&&e[2].data?e[2].dataType===10?e[2].getUint16Array()[0]:e[2].getFloat32Array()[0]:0,a=e[0].dims.length,n=new Int32Array(2*a).fill(0);if(e.length>=4){let u=e[3].getBigInt64Array();for(let d=0;d<u.length;d++)n[Number(u[d])]=Number(r[d]),n[Number(u[d])+a]=Number(r[d+u.length])}else r.forEach((u,d)=>n[Number(d)]=Number(u));let s=[];return n.forEach(u=>s.push(u)),{mode:t.mode,value:i,pads:s}}else return t},zh=(e,t)=>{Nl(e.inputs);let r=Ll(e.inputs,t);e.compute(Wl(e.inputs,r),{inputs:[0]})}}),tr,Yi,Xi,Ji,ea,Vl,Gl,ta,ra,Ch,Ah,ia,Oh,Rh,aa,Bh,Nh,Dh,Mh,C0=P(()=>{Ue(),J(),ie(),ae(),tr=e=>{if(ye.webgpu.validateInputContent&&(!e||e.length!==1))throw new Error("Pool ops requires 1 input.")},Yi=(e,t,r)=>{let i=t.format==="NHWC",a=e.dims.slice();i&&a.splice(1,0,a.pop());let n=Object.hasOwnProperty.call(t,"dilations"),s=t.kernelShape.slice(),u=t.strides.slice(),d=n?t.dilations.slice():[],p=t.pads.slice();Wr.adjustPoolAttributes(r,a,s,u,d,p);let c=Wr.computePoolOutputShape(r,a,u,d,s,p,t.autoPad),f=Object.assign({},t);n?Object.assign(f,{kernelShape:s,strides:u,pads:p,dilations:d,cacheKey:t.cacheKey}):Object.assign(f,{kernelShape:s,strides:u,pads:p,cacheKey:t.cacheKey});let g=c.slice();return g.push(g.splice(1,1)[0]),[f,i?g:c]},Xi=(e,t)=>{let r=t.format==="NHWC",i=B.size(e),a=B.size(t.kernelShape),n=[{type:12,data:i},{type:12,data:a}],s=[{name:"outputSize",type:"u32"},{name:"kernelSize",type:"u32"}];if(t.kernelShape.length<=2){let u=t.kernelShape[t.kernelShape.length-1],d=t.strides[t.strides.length-1],p=t.pads[t.pads.length/2-1],c=t.pads[t.pads.length-1],f=!!(p+c);n.push({type:12,data:u},{type:12,data:d},{type:12,data:p},{type:12,data:c}),s.push({name:"kw",type:"u32"},{name:"sw",type:"u32"},{name:"pwStart",type:"u32"},{name:"pwEnd",type:"u32"});let g=!1;if(t.kernelShape.length===2){let _=t.kernelShape[t.kernelShape.length-2],w=t.strides[t.strides.length-2],b=t.pads[t.pads.length/2-2],S=t.pads[t.pads.length-2];g=!!(b+S),n.push({type:12,data:_},{type:12,data:w},{type:12,data:b},{type:12,data:S}),s.push({name:"kh",type:"u32"},{name:"sh",type:"u32"},{name:"phStart",type:"u32"},{name:"phEnd",type:"u32"})}return[n,s,!0,f,g]}else{if(r)throw new Error("Pooling with kernelShape.length > 2 is not supported for NHWC format.");let u=B.computeStrides(t.kernelShape);n.push({type:12,data:u},{type:12,data:t.pads},{type:12,data:t.strides}),s.push({name:"kernelStrides",type:"u32",length:u.length},{name:"pads",type:"u32",length:t.pads.length},{name:"strides",type:"u32",length:t.strides.length});let d=t.pads.reduce((p,c)=>p+c);return[n,s,!!d,!1,!1]}},Ji=(e,t,r,i,a,n,s,u,d,p,c,f)=>{let g=a.format==="NHWC",_=t.type.value,w=j("output",t.type.tensor,i);if(a.kernelShape.length<=2){let b="",S="",v="",$=r-(g?2:1);if(c?b=`
                for (var i: u32 = 0u; i < uniforms.kw; i++) {
                  xIndices[${$}] = indices[${$}] * uniforms.sw - uniforms.pwStart + i;
                  if (xIndices[${$}] < 0 || xIndices[${$}]
                      >= uniforms.x_shape[${$}]) {
                    pad++;
                    continue;
                  }
                  let x_val = x[${t.indicesToOffset("xIndices")}];
                  ${n}
                }`:b=`
                for (var i: u32 = 0u; i < uniforms.kw; i++) {
                  xIndices[${$}] = indices[${$}] * uniforms.sw - uniforms.pwStart + i;
                  let x_val = x[${t.indicesToOffset("xIndices")}];
                  ${n}
                }`,a.kernelShape.length===2){let I=r-(g?3:2);f?S=`
                for (var j: u32 = 0u; j < uniforms.kh; j++) {
                  xIndices[${I}] = indices[${I}] * uniforms.sh - uniforms.phStart + j;
                  if (xIndices[${I}] < 0 || xIndices[${I}] >= uniforms.x_shape[${I}]) {
                    pad += i32(uniforms.kw);
                    continue;
                  }
              `:S=`
                for (var j: u32 = 0u; j < uniforms.kh; j++) {
                  xIndices[${I}] = indices[${I}] * uniforms.sh - uniforms.phStart + j;
                `,v=`
              }
            `}return`
            ${e.registerUniforms(d).declareVariables(t,w)}

            ${e.mainStart()}
              ${e.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}

              let indices = ${w.offsetToIndices("global_idx")};
              var xIndices = ${w.offsetToIndices("global_idx")};

              var value = ${_}(${u});
              var pad = 0;
              ${S}
              ${b}
              ${v}
              ${s}

              output[global_idx] = value;
            }`}else{if(g)throw new Error("Pooling with kernelShape.length > 2 is not supported for NHWC format.");let b=a.kernelShape.length,S=a.pads.length,v="";return p?v=`
                if (xIndices[j] >= uniforms.x_shape[j]) {
                  pad++;
                  isPad = true;
                  break;
                }
              }
              if (!isPad) {
                let x_val = x[${t.indicesToOffset("xIndices")}];
                ${n}
              }`:v=`
              }
              let x_val = x[${t.indicesToOffset("xIndices")}];
              ${n}
            `,`
            ${e.registerUniforms(d).declareVariables(t,w)}

            ${e.mainStart()}
              ${e.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
              let indices = ${w.offsetToIndices("global_idx")};
              var xIndices = ${w.offsetToIndices("global_idx")};

              var offsets: array<u32, ${b}>;

              var value = ${_}(${u});
              var pad = 0;
              var isPad = false;

              for (var i: u32 = 0u; i < uniforms.kernelSize; i++) {
                var offset = i;
                for (var j = 0u; j < ${b-1}u; j++) {
                  offsets[j] = offset / ${Z("uniforms.kernelStrides","j",b)};
                  offset -= offsets[j] * ${Z("uniforms.kernelStrides","j",b)};
                }
                offsets[${b-1}] = offset;

                isPad = false;
                for (var j = ${r-b}u; j < ${r}u; j++) {
                  xIndices[j] = indices[j] * ${Z("uniforms.strides",`j - ${r-b}u`,b)}
                    + offsets[j - ${r-b}u] - ${Z("uniforms.pads","j - 2u",S)};
                  ${v}
              }
              ${s}

              output[global_idx] = value;
            }`}},ea=e=>`${e.format};${e.ceilMode};${e.autoPad};${e.kernelShape.length}`,Vl=e=>`${ea(e)};${e.countIncludePad}`,Gl=e=>`${ea(e)};${e.storageOrder};${e.dilations}`,ta=e=>({format:e.format,autoPad:["NOTSET","VALID","SAME_UPPER","SAME_LOWER"][e.auto_pad],ceilMode:e.ceil_mode,kernelShape:e.kernel_shape,strides:e.strides,pads:e.pads}),ra=(e,t,r,i)=>{let[a,n]=Yi(t,i,r),s=M("x",t.dataType,t.dims.length),u=s.type.value,d="value += x_val;",p="";a.countIncludePad?p+=`value /= ${u}(uniforms.kernelSize);`:p+=`value /= ${u}(i32(uniforms.kernelSize) - pad);`;let[c,f,g,_,w]=Xi(n,a);c.push(...Q(t.dims,n));let b=["rank"];return{name:e,shaderCache:{hint:`${i.cacheKey};${g};${_};${w}`,inputDependencies:b},getRunData:()=>({outputs:[{dims:n,dataType:t.dataType}],dispatchGroup:{x:Math.ceil(B.size(n)/64)},programUniforms:c}),getShaderSource:S=>Ji(S,s,t.dims.length,n.length,a,d,p,0,f,g,_,w)}},Ch=e=>{let t=e.count_include_pad!==0,r=ta(e);if(r.ceilMode!==0)throw new Error("using ceil() in shape computation is not yet supported for AveragePool");let i={countIncludePad:t,...r,cacheKey:""};return{...i,cacheKey:Vl(i)}},Ah=(e,t)=>{tr(e.inputs),e.compute(ra("AveragePool",e.inputs[0],!1,t))},ia={autoPad:"",ceilMode:0,countIncludePad:!1,kernelShape:[],strides:[],pads:[],storageOrder:0,dilations:[]},Oh=e=>{let t=e.format;return{format:t,...ia,cacheKey:t}},Rh=(e,t)=>{tr(e.inputs),e.compute(ra("GlobalAveragePool",e.inputs[0],!0,t))},aa=(e,t,r,i)=>{let[a,n]=Yi(t,i,r),s=`
      value = max(x_val, value);
    `,u="",d=M("x",t.dataType,t.dims.length),p=["rank"],[c,f,g,_,w]=Xi(n,a);return c.push(...Q(t.dims,n)),{name:e,shaderCache:{hint:`${i.cacheKey};${g};${_};${w}`,inputDependencies:p},getRunData:()=>({outputs:[{dims:n,dataType:t.dataType}],dispatchGroup:{x:Math.ceil(B.size(n)/64)},programUniforms:c}),getShaderSource:b=>Ji(b,d,t.dims.length,n.length,a,s,u,t.dataType===10?-65504:-1e5,f,g,_,w)}},Bh=(e,t)=>{tr(e.inputs),e.compute(aa("MaxPool",e.inputs[0],!1,t))},Nh=e=>{let t=e.storage_order,r=e.dilations,i=ta(e);if(t!==0)throw new Error("column major storage order is not yet supported for MaxPool");if(i.ceilMode!==0)throw new Error("using ceil() in shape computation is not yet supported for MaxPool");let a={storageOrder:t,dilations:r,...i,cacheKey:""};return{...a,cacheKey:Gl(a)}},Dh=e=>{let t=e.format;return{format:t,...ia,cacheKey:t}},Mh=(e,t)=>{tr(e.inputs),e.compute(aa("GlobalMaxPool",e.inputs[0],!0,t))}}),Hl,Fl,Uh,Ph,A0=P(()=>{J(),ie(),ve(),ae(),Hl=(e,t)=>{if(e.length<2||e.length>3)throw new Error("DequantizeLinear requires 2 or 3 inputs.");if(e.length===3&&e[1].dims===e[2].dims)throw new Error("x-scale and x-zero-point must have the same shape.");if(e.length===3&&e[0].dataType!==e[2].dataType)throw new Error("x and x-zero-point must have the same data type.");if(e[0].dataType===6&&e.length>2)throw new Error("In the case of dequantizing int32 there is no zero point.");if(e[1].dims.length!==0&&e[1].dims.length!==1&&e[1].dims.length!==e[0].dims.length)throw new Error("scale input must be a scalar, a 1D tensor, or have the same rank as the input tensor.");if(e.length>2){if(e[0].dataType!==e[2].dataType)throw new Error("x and x-zero-point must have the same data type.");if(e[1].dims.length!==e[2].dims.length)throw new Error("scale and zero-point inputs must have the same rank.");if(!e[1].dims.map((r,i)=>r===e[2].dims[i]).reduce((r,i)=>r&&i,!0))throw new Error("scale and zero-point inputs must have the same shape.")}if(t.blockSize>0){if(e[1].dims.length===0||e[1].dims.length===1&&e[1].dims[0]===1)throw new Error("blockSize must be set only for block quantization.");if(!e[1].dims.map((a,n)=>n===t.axis||a===e[0].dims[n]).reduce((a,n)=>a&&n,!0))throw new Error("For block qunatization, scale input shape to match the input shape except for the axis");if(e[1].dims.length!==e[0].dims.length)throw new Error("For block qunatization the scale input rank must be the same as the x rank.");let r=e[0].dims[t.axis],i=e[1].dims[t.axis];if(t.blockSize<Math.ceil(r/i)||t.blockSize>Math.ceil(r/(i-1)-1))throw new Error("blockSize must be with in the range [ceil(dI / Si), ceil(dI / (Si - 1) - 1)].")}},Fl=(e,t)=>{let r=B.normalizeAxis(t.axis,e[0].dims.length),i=e[0].dataType,a=i===3,n=e[0].dims,s=e[1].dataType,u=B.size(n),d=i===3||i===2,p=d?[Math.ceil(B.size(e[0].dims)/4)]:e[0].dims,c=e[1].dims,f=e.length>2?e[2]:void 0,g=f?d?[Math.ceil(B.size(f.dims)/4)]:f.dims:void 0,_=c.length===0||c.length===1&&c[0]===1,w=_===!1&&c.length===1,b=$e(u),S=_&&(!d||b===4),v=S?b:1,$=S&&!d?b:1,I=M("input",d?12:i,p.length,$),T=M("scale",s,c.length),E=f?M("zero_point",d?12:i,g.length):void 0,A=j("output",s,n.length,v),C=[I,T];E&&C.push(E);let O=[p,c];f&&O.push(g);let U=[{type:12,data:u/v},{type:12,data:r},{type:12,data:t.blockSize},...Q(...O,n)],x=Y=>{let G=[{name:"output_size",type:"u32"},{name:"axis",type:"u32"},{name:"block_size",type:"u32"}];return`
      ${Y.registerUniforms(G).declareVariables(...C,A)}
      ${Y.mainStart()}
          ${Y.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
          let output_indices = ${A.offsetToIndices("global_idx")};

          // Set input x
          ${d?`
            let input = ${I.getByOffset("global_idx / 4")};
            let x_vec = ${a?"unpack4xI8(input)":"unpack4xU8(input)"};
            let x_value = ${v===1?"x_vec[global_idx % 4]":"x_vec"};`:`let x_value = ${I.getByOffset("global_idx")};`};

          // Set scale input
          ${_?`let scale_value= ${T.getByOffset("0")}`:w?`
            let scale_index = ${A.indicesGet("output_indices","uniforms.axis")};
            let scale_value= ${T.getByOffset("scale_index")};`:`
            var scale_indices: ${T.type.indices} = output_indices;
            let index = ${T.indicesGet("scale_indices","uniforms.axis")} / uniforms.block_size;
            ${T.indicesSet("scale_indices","uniforms.axis","index")};
            let scale_value= ${T.getByIndices("scale_indices")};`};

          // Set zero-point input
          ${E?_?d?`
                let zero_point_input = ${E.getByOffset("0")};
                let zero_point_vec =  ${a?"unpack4xI8(zero_point_input)":"unpack4xU8(zero_point_input)"};
                let zero_point_value= zero_point_vec[0]`:`let zero_point_value = ${E.getByOffset("0")}`:w?d?`
                let zero_point_index = ${A.indicesGet("output_indices","uniforms.axis")};
                let zero_point_input = ${E.getByOffset("zero_point_index / 4")};
                let zero_point_vec =  ${a?"unpack4xI8(zero_point_input)":"unpack4xU8(zero_point_input)"};
                let zero_point_value = zero_point_vec[zero_point_index % 4]`:`
                let zero_point_index = ${A.indicesGet("output_indices","uniforms.axis")};
                let zero_point_value = ${E.getByOffset("zero_point_index")};`:d?`
                let zero_point_offset = ${T.indicesToOffset("scale_indices")};
                let zero_point_input = ${E.getByOffset("zero_point_offset / 4")};
                let zero_point_vec = ${a?"unpack4xI8(zero_point_input)":"unpack4xU8(zero_point_input)"};
                let zero_point_value = zero_point_vec[zero_point_offset % 4];`:`let zero_point_value = ${E.getByIndices("scale_indices")};`:`let zero_point_value = ${d?a?"i32":"u32":I.type.value}(0);`};
      // Compute and write output
      ${A.setByOffset("global_idx",`${A.type.value}(x_value - zero_point_value) * scale_value`)};
      }`};return{name:"DequantizeLinear",shaderCache:{hint:t.cacheKey,inputDependencies:E?["rank","rank","rank"]:["rank","rank"]},getShaderSource:x,getRunData:()=>({outputs:[{dims:n,dataType:s}],dispatchGroup:{x:Math.ceil(u/v/64),y:1,z:1},programUniforms:U})}},Uh=(e,t)=>{Hl(e.inputs,t),e.compute(Fl(e.inputs,t))},Ph=e=>ce({axis:e.axis,blockSize:e.blockSize})}),jl,Kl,qh,O0=P(()=>{Ue(),J(),ae(),jl=(e,t,r)=>{let i=e===t,a=e<t&&r<0,n=e>t&&r>0;if(i||a||n)throw new Error("Range these inputs' contents are invalid.")},Kl=(e,t,r,i)=>{let a=Math.abs(Math.ceil((t-e)/r)),n=[a],s=a,u=[{type:12,data:s},{type:i,data:e},{type:i,data:r},...Q(n)],d=p=>{let c=j("output",i,n.length),f=c.type.value,g=[{name:"outputSize",type:"u32"},{name:"start",type:f},{name:"delta",type:f}];return`
        ${p.registerUniforms(g).declareVariables(c)}
        ${p.mainStart()}
        ${p.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
        output[global_idx] = uniforms.start + ${f}(global_idx) * uniforms.delta;
      }`};return{name:"Range",shaderCache:{hint:`${i}`},getShaderSource:d,getRunData:()=>({outputs:[{dims:n,dataType:i}],dispatchGroup:{x:Math.ceil(s/64)},programUniforms:u})}},qh=e=>{let t=0,r=0,i=0;e.inputs[0].dataType===6?(t=e.inputs[0].getInt32Array()[0],r=e.inputs[1].getInt32Array()[0],i=e.inputs[2].getInt32Array()[0]):e.inputs[0].dataType===1&&(t=e.inputs[0].getFloat32Array()[0],r=e.inputs[1].getFloat32Array()[0],i=e.inputs[2].getFloat32Array()[0]),ye.webgpu.validateInputContent&&jl(t,r,i),e.compute(Kl(t,r,i,e.inputs[0].dataType),{inputs:[]})}}),Zl,Ql,Wh,Lh,R0=P(()=>{J(),ie(),ve(),ae(),Zl=(e,t,r,i)=>{if(e!=="none"&&i!=="i32"&&i!=="u32"&&i!=="f32")throw new Error(`Input ${i} is not supported with reduction ${e}.`);let a=`{
                var oldValue = 0;
                loop {
                  let newValueF32 =`,n=`;
                  let newValue = bitcast<i32>(newValueF32);
                  let res = atomicCompareExchangeWeak(&${t}, oldValue, newValue);
                  if res.exchanged {
                    break;
                  }
                  oldValue = res.old_value;
                }
              }`;switch(e){case"none":return`${t}=${r};`;case"add":return i==="i32"||i==="u32"?`atomicAdd(&${t}, bitcast<${i}>(${r}));`:`
              ${a}bitcast<${i}>(oldValue) + (${r})${n}`;case"max":return i==="i32"||i==="u32"?`atomicMax(&${t}, bitcast<${i}>(${r}));`:`
                ${a}max(bitcast<f32>(oldValue), (${r}))${n}`;case"min":return i==="i32"||i==="u32"?`atomicMin(&${t}, bitcast<${i}>(${r}));`:`${a}min(bitcast<${i}>(oldValue), (${r}))${n}`;case"mul":return`${a}(bitcast<${i}>(oldValue) * (${r}))${n}`;default:throw new Error(`Reduction ${e} is not supported.`)}},Ql=(e,t)=>{let r=e[0].dims,i=e[1].dims,a=r,n=1,s=Math.ceil(B.sizeToDimension(i,i.length-1)/n),u=i[i.length-1],d=B.sizeFromDimension(r,u),p=[{type:12,data:s},{type:12,data:u},{type:12,data:d},...Q(e[1].dims,e[2].dims,a)],c=f=>{let g=M("indices",e[1].dataType,e[1].dims.length),_=M("updates",e[2].dataType,e[2].dims.length,n),w=t.reduction!=="none"&&t.reduction!==""?gp("output",e[0].dataType,a.length):j("output",e[0].dataType,a.length,n);return`
      ${f.registerUniform("output_size","u32").registerUniform("last_index_dimension","u32").registerUniform("num_updates_elements","u32").declareVariables(g,_,w)}
      ${f.mainStart()}
        ${f.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
  var data_offset = 0u;
  let indices_start = uniforms.last_index_dimension * global_idx;
  let indices_end = indices_start + uniforms.last_index_dimension;
  for (var i = indices_start; i < indices_end; i++) {
    var index = i32(indices[i].x);
    ${e[0].dims.length===1?`
    let element_count_dim = uniforms.output_strides;
    let dim_value = uniforms.output_shape;`:`
    let element_count_dim = uniforms.output_strides[i - indices_start];
    let dim_value = uniforms.output_shape[i - indices_start];`}
    if (index >= 0) {
      if (index >= i32(dim_value)) {
        index = i32(dim_value - 1);
      }
    } else {
      if (index < -i32(dim_value)) {
        index = 0;
      } else {
        index += i32(dim_value);
      }
    }
    data_offset += u32((u32(index) * element_count_dim));
  }

  for (var i = 0u; i < uniforms.num_updates_elements; i++) {
    let value = updates[uniforms.num_updates_elements * global_idx + i];
    ${Zl(t.reduction,"output[data_offset + i]","value",w.type.value)}
  }

      }`};return{name:"ScatterND",shaderCache:{hint:`${t.cacheKey}_${t.reduction}`,inputDependencies:["rank","rank"]},getRunData:()=>({outputs:[{dims:a,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(s/64)},programUniforms:p}),getShaderSource:c}},Wh=e=>ce({reduction:e.reduction}),Lh=(e,t)=>{e.compute(Ql(e.inputs,t),{inputs:[e.inputs[1],e.inputs[2]],outputs:[]})}}),Yl,Xl,Jl,na,ed,td,rd,id,ad,nd,sd,od,sa,ud,ld,dd,pd,cd,Vh,Gh,B0=P(()=>{J(),ie(),ve(),ae(),Yl=(e,t)=>{if(e.every(r=>r>0||(()=>{throw new Error("Resize requires scales input values to be positive")})),e.length>0){if(t.mode==="linear"){if(!(e.length===2||e.length===3||e.length===4&&e[0]===1&&e[1]===1||e.length===4&&e[0]===1&&e[3]===1||e.length===5&&e[0]===1&&e[1]===1))throw new Error(`For linear mode, Resize requires scales to be 2D, 3D, 4D with either two outermost or one innermost and
            one outermost scale values equal to 1, or 5D with two outermost scale values equal to 1`)}else if(t.mode==="cubic"&&!(e.length===2||e.length===4&&e[0]===1&&e[1]===1||e.length===4&&e[0]===1&&e[3]===1))throw new Error("Resize requires scales input size to be 2 or 4 for cubic mode")}},Xl=(e,t,r)=>{t.every(a=>a>=0&&a<r||(()=>{throw new Error("Resize requires axes input values to be positive and less than rank")}));let i=new Array(r).fill(1);return t.forEach((a,n)=>i[a]=e[n]),i},Jl=(e,t,r,i,a,n)=>{let[s,u,d]=r>10?[1,2,3]:[-1,e.length>1?1:-1,-1],p=e[0].dims.length;if(s>0&&e.length>s&&e[s].dims.length>0)e[s].getFloat32Array().forEach(c=>n.push(c));else if(t.coordinateTransformMode==="tf_crop_and_resize")throw new Error("Resize requires RoI input to be specified when coordinateTransformMode is tfCropAndResize");if(u>0&&e.length>u&&e[u].dims.length===1&&e[u].dims[0]>0){if(e[u].getFloat32Array().forEach(c=>i.push(c)),i.length!==0&&i.length!==p&&r>=18&&i.length!==t.axes.length)throw new Error("Resize requires scales input size to be same as input rank or axes size for opset 18 and up");Yl(i,t),t.axes.length>0&&Xl(i,t.axes,p).forEach((c,f)=>i[f]=c)}if(d>0&&e.length>d&&e[d].dims.length===1&&e[d].dims[0]>0&&(e[d].getBigInt64Array().forEach(c=>a.push(Number(c))),a.length!==0&&a.length!==p&&r>=18&&a.length!==t.axes.length))throw new Error("Resize requires sizes input size to be same as input rank or axes size for opset 18 and up");if(t.axes.length>0){if(i.length!==0&&i.length!==t.axes.length)throw new Error('Resize requires "scales" input size to be of axes rank when axes attributes is specified');if(a.length!==0&&a.length!==t.axes.length)throw new Error('Resize requires "sizes" input size to be of rank axes rank when axes attributes is specified')}if(typeof i<"u"&&typeof a<"u"&&i.length>0&&a.length>p)throw new Error("Resize requires only of scales or sizes to be specified")},na=(e,t,r,i)=>`
  // The whole part and the fractional part are calculated separately due to inaccuracy of floating
  // point division. As an example, f32(21) / f32(7) may evaluate to 2.99... instead of 3, causing an
  // offset-by-one error later in floor().
  let big = (${e}) * (${t});
  let whole = ${i}(big / (${r}));
  let fract = ${i}(big % (${r})) / ${i}(${r});
  return whole + fract;
`,ed=(e,t)=>`fn getOriginalCoordinateFromResizedCoordinate(xResized: u32, xScale: f32, lengthResized: u32,
     lengthOriginal: u32, roiStart: f32, roiEnd: f32) -> ${t} { `+(()=>{switch(e){case"asymmetric":return`
          if (xScale < 1.0 || floor(xScale) != xScale) {
            return ${t}(xResized) / ${t}(xScale);
          } else {
            ${na("xResized","lengthOriginal","lengthResized",t)}
          }
        `;case"pytorch_half_pixel":return`if (lengthResized > 1) {
                    return (${t}(xResized) + 0.5) / ${t}(xScale) - 0.5;
                  } else {
                    return 0.0;
                  }`;case"tf_half_pixel_for_nn":return`return (${t}(xResized) + 0.5) / ${t}(xScale);`;case"align_corners":return`if (lengthResized == 1) {
                    return 0.0;
                  } else {
                    ${na("xResized","lengthOriginal - 1","lengthResized - 1",t)}
                  }`;case"tf_crop_and_resize":return`if (lengthResized > 1) {
                    return ${t}(roiStart) * ${t}(lengthOriginal - 1) +
                        (${t}(xResized) * ${t}(roiEnd - roiStart) * ${t}(lengthOriginal - 1)) /
                        ${t}(lengthResized - 1);
                  } else {
                    return 0.5 * ${t}(roiStart + roiEnd) * ${t}(lengthOriginal - 1);
                  }`;case"half_pixel_symmetric":return`const outputWidth = ${t}xScale * ${t}(lengthResized);
                  const adjustment = ${t}(lengthResized) / outputWidth;
                  const center = ${t}(lengthOriginal) / 2;
                  const offset = center * (1 - adjustment);
                  return offset + ((${t}(xResized) + 0.5) / ${t}(xScale)) - 0.5;`;case"half_pixel":return`return ((${t}(xResized) + 0.5) / ${t}(xScale)) - 0.5;`;default:throw new Error(`Coordinate transform mode ${e} is not supported`)}})()+"}",td=(e,t,r)=>`fn getNearestPixelFromOriginal(xOriginal: ${r}, isDownSample: bool) -> ${r} {`+(()=>{switch(e){case"round_prefer_ceil":return"if (fract(xOriginal) == 0.5) {             return ceil(xOriginal);           } else {             return round(xOriginal);           }";case"floor":return"return floor(xOriginal);";case"ceil":return"return ceil(xOriginal);";case"round_prefer_floor":return"if (fract(xOriginal) == 0.5) {                     return floor(xOriginal);                   } else {                     return round(xOriginal);                   }";case"simple":default:if(t<11)return"if (isDownSample)                     {                       return ceil(xOriginal);                     } else {                       return xOriginal;                     }";throw new Error(`Nearest mode ${e} is not supported`)}})()+"}",rd=(e,t,r)=>{let i=new Array(r).fill(0).concat(new Array(r).fill(1)),a=e.length===0?i:e.slice();return t.length>0?(t.forEach((n,s)=>{i[n]=a[s],i[s+r]=a[t.length+s]}),i):a},id=(e,t,r,i)=>{let a=[];if(r.length>0)if(i.length>0){if(e.forEach(n=>a.push(n)),Math.max(...i)>e.length)throw new Error("axes is out of bound");i.forEach((n,s)=>a[n]=r[s])}else r.forEach(n=>a.push(n));else{if(t.length===0)throw new Error("Resize requires either scales or sizes.");a=e.map((n,s)=>Math.round(n*t[s]))}return a},ad=(e,t,r)=>{let i=(()=>{switch(r.keepAspectRatioPolicy){case"not_larger":return r.axes.length>0?Math.min(...r.axes.map(n=>t[n]),Number.MAX_VALUE):Math.min(...t,Number.MAX_VALUE);case"not_smaller":return r.axes.length>0?Math.max(...r.axes.map(n=>t[n]),Number.MIN_VALUE):Math.max(...t,Number.MIN_VALUE);default:throw new Error(`Keep aspect ratio policy ${r.keepAspectRatioPolicy} is not supported`)}})();t.fill(1,0,t.length);let a=e.slice();return r.axes.length>0?(r.axes.forEach(n=>t[n]=i),r.axes.forEach(n=>a[n]=Math.round(e[n]*t[n]))):(t.fill(i,0,t.length),a.forEach((n,s)=>a[s]=Math.round(n*t[s]))),a},nd=(e,t,r,i,a)=>`
    fn calculateOriginalIndicesFromOutputIndices(output_indices: ${e.type.indices}) -> array<${e.type.value}, ${r.length}> {
      var original_indices: array<${e.type.value}, ${r.length}>;
      for (var i:u32 = 0; i < ${r.length}; i++) {
        var output_index = ${e.indicesGet("output_indices","i")};
        var scale = ${Z("uniforms.scales","i",i)};
        var roi_low = ${Z("uniforms.roi","i",a)};
        var roi_hi = ${Z("uniforms.roi",`i + ${t.length}`,a)};
        if (scale == 1.0) {
          original_indices[i] = ${e.type.value}(output_index);
        } else {
          var input_shape_i = ${Z("uniforms.input_shape","i",t.length)};
          var output_shape_i = ${Z("uniforms.output_shape","i",r.length)};
          original_indices[i] = getOriginalCoordinateFromResizedCoordinate(output_index, scale, output_shape_i,
                                                                           input_shape_i, roi_low, roi_hi);
        }
      }
      return original_indices;
    }`,sd=(e,t,r,i,a,n,s)=>`
    fn calculateInputIndicesFromOutputIndices(output_indices: ${t.type.indices}) -> ${e.type.indices} {
      var input_indices: ${e.type.indices};
      for (var i:u32 = 0; i < ${i.length}; i++) {
        var output_index = ${t.indicesGet("output_indices","i")};
        var input_index: u32;
        var scale = ${Z("uniforms.scales","i",a)};
        if (scale == 1.0) {
          input_index = output_index;
        } else {
          var roi_low = ${Z("uniforms.roi","i",n)};
          var roi_hi = ${Z("uniforms.roi",`i + ${r.length}`,n)};
          var input_shape_i = ${Z("uniforms.input_shape","i",r.length)};
          var output_shape_i = ${Z("uniforms.output_shape","i",i.length)};
          var original_idx = getOriginalCoordinateFromResizedCoordinate(output_index, scale, output_shape_i,
                                                                        input_shape_i, roi_low, roi_hi);
          if (!${s} || (original_idx >= 0 && original_idx < ${t.type.value}(input_shape_i))) {
            if (original_idx < 0) {
              input_index = 0;
            } else if (original_idx > ${t.type.value}(input_shape_i - 1)) {
              input_index = input_shape_i - 1;
            } else {
              input_index = u32(getNearestPixelFromOriginal(original_idx, scale < 1));
            }
          } else {
            input_index = u32(original_idx);
          }
        }
        ${e.indicesSet("input_indices","i","input_index")}
      }
      return input_indices;
    }`,od=(e,t)=>`
    fn checkInputIndices(input_indices: ${e.type.indices}) -> bool {
      for (var i:u32 = 0; i < ${t.length}; i++) {
        var input_index = ${e.indicesGet("input_indices","i")};
        if (input_index < 0 || input_index >= ${Z("uniforms.input_shape","i",t.length)}) {
          return false;
        }
      }
      return true;
    }`,sa=(e,t,r,i)=>e.rank>i?`
    ${e.indicesSet("input_indices",t,"channel")};
    ${e.indicesSet("input_indices",r,"batch")};
`:"",ud=(e,t,r,i,a)=>{let[n,s,u,d]=r.length===2?[-1,0,1,-1]:[0,2,3,1],p=e.type.value;return`
    fn getInputValue(batch: u32, channel: u32, row: u32, col: u32) -> ${p} {
      var input_indices: ${e.type.indices};
      ${e.indicesSet("input_indices",s,`max(0, min(row, ${r[s]} - 1))`)};
      ${e.indicesSet("input_indices",u,`max(0, min(col, ${r[u]} - 1))`)};
      ${sa(e,d,n,2)}
      return ${e.getByIndices("input_indices")};
    }

    fn bilinearInterpolation(output_indices: ${t.type.indices}) -> ${p} {
      var originalIndices = calculateOriginalIndicesFromOutputIndices(output_indices);
      var row:${p} = originalIndices[${s}];
      var col:${p} = originalIndices[${u}];
      ${i?`if (row < 0 || row > (${r[s]} - 1) || col < 0 || col > (${r[u]} - 1)) {
        return ${a};
      }`:""};
      row = max(0, min(row, ${r[s]} - 1));
      col = max(0, min(col, ${r[u]} - 1));
      var row1: u32 = u32(row);
      var col1: u32 = u32(col);
      var row2: u32 = u32(row + 1);
      var col2: u32 = u32(col + 1);
      var channel: u32 = ${r.length>2?`u32(originalIndices[${d}])`:"0"};
      var batch: u32 =  ${r.length>2?`u32(originalIndices[${n}])`:"0"};
      var x11: ${p} = getInputValue(batch, channel, row1, col1);
      var x12: ${p} = getInputValue(batch, channel, row1, col2);
      var x21: ${p} = getInputValue(batch, channel, row2, col1);
      var x22: ${p} = getInputValue(batch, channel, row2, col2);
      var dx1: ${p} = abs(row - ${p}(row1));
      var dx2: ${p} = abs(${p}(row2) - row);
      var dy1: ${p} = abs(col - ${p}(col1));
      var dy2: ${p} = abs(${p}(col2) - col);
      if (row1 == row2) {
        dx1 = 0.5;
        dx2 = 0.5;
      }
      if (col1 == col2) {
        dy1 = 0.5;
        dy2 = 0.5;
      }
      return (x11 * dx2 * dy2 + x12 * dx2 * dy1 + x21 * dx1 * dy2 + x22 * dx1 * dy1);
    }`},ld=(e,t,r,i,a,n,s,u,d,p)=>{let c=r.length===2,[f,g]=c?[0,1]:[2,3],_=e.type.value,w=b=>{let S=b===f?"row":"col";return`
      fn ${S}CubicInterpolation(input_indices: ${e.type.indices}, output_indices: ${t.type.indices}) -> ${_} {
        var output_index = ${t.indicesGet("output_indices",b)};
        var originalIdx: ${_} = getOriginalCoordinateFromResizedCoordinate(output_index, ${a[b]},
        ${i[b]}, ${r[b]}, ${n[b]}, ${n[b]} + ${r.length});
        var fractOriginalIdx: ${_} = originalIdx - floor(originalIdx);
        var coefs = getCubicInterpolationCoefs(fractOriginalIdx);

        if (${u} && (originalIdx < 0 || originalIdx > (${r[b]} - 1))) {
          return ${d};
        }
        var data: array<${_}, 4> = array<${_}, 4>(0.0, 0.0, 0.0, 0.0);
        for (var i: i32 = -1; i < 3; i++) {
          var ${S}: ${_} = originalIdx + ${_}(i);
          if (${S} < 0 || ${S} >= ${r[b]}) {
            ${p?`coefs[i + 1] = 0.0;
                        continue;`:u?`return ${d};`:`${S} = max(0, min(${S}, ${r[b]} - 1));`};
          }
        var input_indices_copy: ${e.type.indices} = input_indices;
          ${e.indicesSet("input_indices_copy",b,`u32(${S})`)};
          data[i + 1] = ${b===f?e.getByIndices("input_indices_copy"):"rowCubicInterpolation(input_indices_copy, output_indices)"};
        }
        return cubicInterpolation1D(data, coefs);
      }`};return`
    ${w(f)};
    ${w(g)};
  fn getCubicInterpolationCoefs(s: ${_}) -> array<${_}, 4> {
    var absS = abs(s);
    var coeffs: array<${_}, 4> = array<${_}, 4>(0.0, 0.0, 0.0, 0.0);
    var oneMinusAbsS: ${_} = 1.0 - absS;
    var twoMinusAbsS: ${_} = 2.0 - absS;
    var onePlusAbsS: ${_} = 1.0 + absS;
    coeffs[0] = ((${s} * onePlusAbsS - 5 * ${s}) * onePlusAbsS + 8 * ${s}) * onePlusAbsS - 4 * ${s};
    coeffs[1] = ((${s} + 2) * absS - (${s} + 3)) * absS * absS + 1;
    coeffs[2] = ((${s} + 2) * oneMinusAbsS - (${s} + 3)) * oneMinusAbsS * oneMinusAbsS + 1;
    coeffs[3] = ((${s} * twoMinusAbsS - 5 * ${s}) * twoMinusAbsS + 8 * ${s}) * twoMinusAbsS - 4 * ${s};
    return coeffs;
  }

  fn cubicInterpolation1D(x: array<${_}, 4>, coefs: array<${_}, 4>) -> ${_} {
    var coefsSum: ${_} = coefs[0] + coefs[1] + coefs[2] + coefs[3];
    return (x[0] * coefs[0] + x[1] * coefs[1]+ x[2] * coefs[2]+ x[3] * coefs[3]) / coefsSum;
  }

  fn bicubicInterpolation(output_indices: ${t.type.indices}) -> ${_} {
    var input_indices: ${e.type.indices} = output_indices;
    return colCubicInterpolation(input_indices, output_indices);
  }
    `},dd=(e,t,r,i,a)=>{let[n,s,u,d,p]=r.length===3?[-1,0,1,2,-1]:[0,2,3,4,1],c=e.type.value;return`
    fn getInputValue(batch: u32, channel: u32, depth:u32, height: u32, width: u32) -> ${c} {
      var input_indices: ${e.type.indices};
      ${e.indicesSet("input_indices",s,`max(0, min(depth, ${r[s]} - 1))`)};
      ${e.indicesSet("input_indices",u,`max(0, min(height, ${r[u]} - 1))`)};
      ${e.indicesSet("input_indices",d,`max(0, min(width, ${r[d]} - 1))`)};
      ${sa(e,p,n,3)}
      return ${e.getByIndices("input_indices")};
    }

    fn trilinearInterpolation(output_indices: ${t.type.indices}) -> ${c} {
      var originalIndices = calculateOriginalIndicesFromOutputIndices(output_indices);
      var depth:${c} = originalIndices[${s}];
      var height:${c} = originalIndices[${u}];
      var width:${c} = originalIndices[${d}];
      ${i?`if (depth < 0 || depth > (${r[s]} - 1) || height < 0 || height > (${r[u]} - 1) || width < 0 || (width > ${r[d]} - 1)) {
      return ${a};
        }`:""};

    depth = max(0, min(depth, ${r[s]} - 1));
      height = max(0, min(height, ${r[u]} - 1));
      width = max(0, min(width, ${r[d]} - 1));
      var depth1: u32 = u32(depth);
      var height1: u32 = u32(height);
      var width1: u32 = u32(width);
      var depth2: u32 = u32(depth + 1);
      var height2: u32 = u32(height + 1);
      var width2: u32 = u32(width + 1);
      var channel: u32 = ${r.length>3?`u32(originalIndices[${p}])`:"0"};
      var batch: u32 =  ${r.length>3?`u32(originalIndices[${n}])`:"0"};

      var x111: ${c} = getInputValue(batch, channel, depth1, height1, width1);
      var x112: ${c} = getInputValue(batch, channel, depth1, height1, width2);
      var x121: ${c} = getInputValue(batch, channel, depth1, height2, width1);
      var x122: ${c} = getInputValue(batch, channel, depth1, height2, width2);
      var x211: ${c} = getInputValue(batch, channel, depth2, height1, width1);
      var x212: ${c} = getInputValue(batch, channel, depth2, height1, width2);
      var x221: ${c} = getInputValue(batch, channel, depth2, height2, width1);
      var x222: ${c} = getInputValue(batch, channel, depth2, height2, width2);
      var dx1: ${c} = abs(depth - ${c}(depth1));
      var dx2: ${c} = abs(${c}(depth2) - depth);
      var dy1: ${c} = abs(height - ${c}(height1));
      var dy2: ${c} = abs(${c}(height2) - height);
      var dz1: ${c} = abs(width - ${c}(width1));
      var dz2: ${c} = abs(${c}(width2) - width);
      if (depth1 == depth2) {
        dx1 = 0.5;
        dx2 = 0.5;
      }
      if (height1 == height2) {
        dy1 = 0.5;
        dy2 = 0.5;
      }
      if (width1 == width2) {
        dz1 = 0.5;
        dz2 = 0.5;
      }
      return (x111 * dx2 * dy2 * dz2 + x112 * dx2 * dy2 * dz1 + x121 * dx2 * dy1 *dz2 + x122 * dx2 * dy1 * dz1 +
              x211 * dx1 * dy2 * dz2 + x212 * dx1 * dy2 * dz1 + x221 * dx1 * dy1 *dz2 + x222 * dx1 * dy1 * dz1);
    }`},pd=(e,t,r,i,a,n)=>{let s=e.dims,u=rd(n,t.axes,s.length),d=id(s,i,a,t.axes),p=i.slice();i.length===0&&(p=s.map(($,I)=>$===0?1:d[I]/$),t.keepAspectRatioPolicy!=="stretch"&&(d=ad(s,p,t)));let c=j("output",e.dataType,d.length),f=M("input",e.dataType,s.length),g=B.size(d),_=s.length===d.length&&s.every(($,I)=>$===d[I]),w=t.coordinateTransformMode==="tf_crop_and_resize",b=t.extrapolationValue,S=f.type.value,v=$=>`
      ${_?"":`
      ${ed(t.coordinateTransformMode,S)};
      ${(()=>{switch(t.mode){case"nearest":return`
              ${od(f,s)};
              ${td(t.nearestMode,r,S)};
              ${sd(f,c,s,d,p.length,u.length,w)};
              `;case"linear":return`
              ${nd(c,s,d,p.length,u.length)};
              ${(()=>{if(s.length===2||s.length===4)return`${ud(f,c,s,w,b)}`;if(s.length===3||s.length===5)return`${dd(f,c,s,w,b)}`;throw Error("Linear mode only supports input dims 2, 3, 4 and 5 are supported in linear mode.")})()};
            `;case"cubic":return`
            ${(()=>{if(s.length===2||s.length===4)return`${ld(f,c,s,d,p,u,t.cubicCoeffA,w,t.extrapolationValue,t.excludeOutside)}`;throw Error("Cubic mode only supports input dims 2 and 4 are supported in linear mode.")})()};
            `;default:throw Error("Invalid resize mode")}})()};
      `}
      ${$.registerUniform("output_size","u32").registerUniform("scales","f32",p.length).registerUniform("roi","f32",u.length).declareVariables(f,c)}
      ${$.mainStart()}
        ${$.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
        ${_?"output[global_idx] = input[global_idx];":`
        let output_indices = ${c.offsetToIndices("global_idx")};
        var input_indices: ${f.type.indices};
        ${(()=>{switch(t.mode){case"nearest":return`input_indices = calculateInputIndicesFromOutputIndices(output_indices);
                if (checkInputIndices(input_indices)) {
                  output[global_idx] = ${f.getByIndices("input_indices")};
                } else {
                  output[global_idx] = ${t.extrapolationValue};
                }`;case"linear":return`output[global_idx] = ${s.length===2||s.length===4?"bilinearInterpolation":"trilinearInterpolation"}(output_indices);`;case"cubic":return"output[global_idx] = bicubicInterpolation(output_indices);";default:throw Error(`Unsupported resize mode: ${t.mode}`)}})()};
`}
      }`;return{name:"Resize",shaderCache:{hint:`${t.cacheKey}|${r}|${p.length>0?t.mode==="cubic"?p:p.length:""}|${a.length>0?a:""}|${u.length>0?u:""}|${_}|${t.mode==="nearest"?s.length:s}`,inputDependencies:["rank"]},getShaderSource:v,getRunData:()=>({outputs:[{dims:d,dataType:e.dataType}],dispatchGroup:{x:Math.ceil(g/64)},programUniforms:[{type:12,data:g},{type:1,data:p},{type:1,data:u},...Q(s,d)]})}},cd=e=>{let t=e.customDataBuffer;return new Uint32Array(t,t.byteOffset,1)[0]},Vh=(e,t)=>{let r=[],i=[],a=[],n=cd(e);if(t.antialias!==0)throw Error("Only default value (0) for Antialias attribute is supported");Jl(e.inputs,t,n,r,i,a),e.compute(pd(e.inputs[0],t,n,r,i,a),{inputs:[0]})},Gh=e=>{let t=e.antialias,r=e.axes,i=e.coordinateTransformMode,a=e.cubicCoeffA,n=e.excludeOutside!==0,s=e.extrapolationValue,u=e.keepAspectRatioPolicy,d=e.mode,p=e.nearestMode===""?"simple":e.nearestMode;return ce({antialias:t,axes:r,coordinateTransformMode:i,cubicCoeffA:a,excludeOutside:n,extrapolationValue:s,keepAspectRatioPolicy:u,mode:d,nearestMode:p})}}),hd,fd,Hh,N0=P(()=>{J(),ie(),ae(),hd=e=>{if(!e||e.length<3)throw new Error("layerNorm requires at least 3 inputs.");let t=e[0],r=e[1],i=e[2];if(t.dataType!==r.dataType||t.dataType!==i.dataType)throw new Error("All inputs must have the same data type");if(t.dims.length!==3&&t.dims.length!==2)throw new Error("Input must be 2D or 3D");if(r.dims.length!==3&&r.dims.length!==2)throw new Error("Skip must be 2D or 3D");let a=t.dims[t.dims.length-1],n=t.dims[t.dims.length-2];if(r.dims[r.dims.length-1]!==a)throw new Error("Skip must have the same hidden size as input");if(r.dims[r.dims.length-2]!==n)throw new Error("Skip must have the same sequence length as input");if(i.dims.length!==1)throw new Error("Gamma must be 1D");if(i.dims[i.dims.length-1]!==a)throw new Error("Gamma must have the same hidden size as input");if(e.length>3){let s=e[3];if(s.dims.length!==1)throw new Error("Beta must be 1D");if(s.dims[s.dims.length-1]!==a)throw new Error("Beta must have the same hidden size as input")}if(e.length>4){let s=e[4];if(s.dims.length!==1)throw new Error("Bias must be 1D");if(s.dims[s.dims.length-1]!==a)throw new Error("Bias must have the same hidden size as input")}},fd=(e,t,r,i)=>{let a=t.simplified,n=e[0].dims,s=B.size(n),u=n,d=s,p=n.slice(-1)[0],c=i?n.slice(0,-1).concat(1):[],f=!a&&e.length>3,g=e.length>4,_=i&&r>1,w=i&&r>2,b=r>3,S=64,v=$e(p),$=[{type:12,data:d},{type:12,data:v},{type:12,data:p},{type:1,data:t.epsilon}],I=E=>{let A=[{name:"output_size",type:"u32"},{name:"components",type:"u32"},{name:"hidden_size",type:"u32"},{name:"epsilon",type:"f32"}],C=[M("x",e[0].dataType,e[0].dims,v),M("skip",e[1].dataType,e[1].dims,v),M("gamma",e[2].dataType,e[2].dims,v)];f&&C.push(M("beta",e[3].dataType,e[3].dims,v)),g&&C.push(M("bias",e[4].dataType,e[4].dims,v)),C.push(j("output",e[0].dataType,u,v)),_&&C.push(j("mean_output",1,c)),w&&C.push(j("inv_std_output",1,c)),b&&C.push(j("input_skip_bias_sum",e[0].dataType,u,v));let O=ke(e[0].dataType),U=ke(1,v);return`

      ${E.registerUniforms(A).declareVariables(...C)}
      var<workgroup> sum_shared : array<${U}, ${S}>;
      var<workgroup> sum_squared_shared : array<${U}, ${S}>;

      ${E.mainStart([S,1,1])}
        let ix = local_id.x;
        let iy = global_id.x / ${S};

        let hidden_size_vectorized: u32 = uniforms.hidden_size / uniforms.components;
        var stride = hidden_size_vectorized / ${S};
        let offset = ix * stride + iy * hidden_size_vectorized;
        let offset1d = stride * ix;
        if (ix == ${S-1}) {
          stride = hidden_size_vectorized - stride * ix;
        }
        for (var i: u32 = 0; i < stride; i++) {
          let skip_value = skip[offset + i];
          let bias_value = ${g?"bias[offset1d + i]":O+"(0.0)"};
          let input_value = x[offset + i];
          let value = input_value + skip_value + bias_value;
          ${b?"input_skip_bias_sum[offset + i] = value;":""}
          output[offset + i] = value;
          let f32_value = ${qt(O,v,"value")};
          sum_shared[ix] += f32_value;
          sum_squared_shared[ix] += f32_value * f32_value;
        }
        workgroupBarrier();

        var reduce_size : u32 = ${S};
        for (var curr_size = reduce_size >> 1;  curr_size > 0; curr_size = reduce_size >> 1) {
          reduce_size = curr_size + (reduce_size & 1);
          if (ix < curr_size) {
            sum_shared[ix] += sum_shared[ix + reduce_size];
            sum_squared_shared[ix] += sum_squared_shared[ix + reduce_size];
          }
          workgroupBarrier();
        }

        let sum = sum_shared[0];
        let square_sum = sum_squared_shared[0];
        let mean = ${gt("sum",v)} / f32(uniforms.hidden_size);
        let inv_std_dev = inverseSqrt(${gt("square_sum",v)} / f32(uniforms.hidden_size) ${a?"":"- mean * mean"} + uniforms.epsilon);
        ${_?"mean_output[global_idx] = mean;":""}
        ${w?"inv_std_output[global_idx] = inv_std_dev;":""}

        for (var i: u32 = 0; i < stride; i++) {
          output[offset + i] = (output[offset + i] ${a?"":`- ${O}(mean)`}) *
            ${O}(inv_std_dev) * gamma[offset1d + i]
            ${f?"+ beta[offset1d + i]":""};
        }
      }`},T=[{dims:u,dataType:e[0].dataType}];return r>1&&T.push({dims:c,dataType:1}),r>2&&T.push({dims:c,dataType:1}),r>3&&T.push({dims:n,dataType:e[0].dataType}),{name:"SkipLayerNormalization",shaderCache:{hint:`${v};${_};${w};${b}`,inputDependencies:e.map((E,A)=>"type")},getShaderSource:I,getRunData:()=>({outputs:T,dispatchGroup:{x:Math.ceil(d/p)},programUniforms:$})}},Hh=(e,t)=>{hd(e.inputs);let r=[0];e.outputCount>1&&r.push(-3),e.outputCount>2&&r.push(-3),e.outputCount>3&&r.push(3),e.compute(fd(e.inputs,t,e.outputCount,!1),{outputs:r})}}),md,rr,gd,oa,yd,_d,Fh,jh,D0=P(()=>{J(),ie(),ve(),ae(),md=(e,t)=>{if(!e||e.length<1)throw new Error("too few inputs");if(t.axes.length!==0){if(t.axes.length!==t.starts.length||t.axes.length!==t.ends.length)throw new Error("axes, starts and ends must have the same length")}else if(t.starts.length!==t.ends.length)throw new Error("starts and ends must have the same length");e.slice(1).forEach((r,i)=>{if(e[i+1].dataType!==6&&e[i+1].dataType!==7)throw new Error(`Input ${i} must be an array of int32 or int64`)})},rr=(e,t)=>{let r=[];if(e.length>t)if(e[t].dataType===7)e[t].getBigInt64Array().forEach(i=>r.push(Number(i)));else if(e[t].dataType===6)e[t].getInt32Array().forEach(i=>r.push(Number(i)));else throw new Error(`Input ${t} must be an array of int32 or int64`);return r},gd=(e,t)=>{if(e.length>1){let r=rr(e,1),i=rr(e,2),a=rr(e,3);return a.length===0&&(a=[...Array(e[0].dims.length).keys()]),ce({starts:r,ends:i,axes:a})}else return t},oa=(e,t,r,i,a)=>{let n=e;return e<0&&(n+=r[i[t]]),a[t]<0?Math.max(0,Math.min(n,r[i[t]]-1)):Math.max(0,Math.min(n,r[i[t]]))},yd=(e,t,r)=>`fn calculateInputIndices(output_indices: ${t.type.indices}) -> ${e.type.indices} {
          var input_indices: ${e.type.indices};
          var carry = 0u;
          for (var i = ${r.length-1}; i >= 0; i--) {
            let input_shape_i = ${Z("uniforms.input_shape","i",r.length)};
            let steps_i = ${Z("uniforms.steps","i",r.length)};
            let signs_i = ${Z("uniforms.signs","i",r.length)};
            let starts_i = ${Z("uniforms.starts","i",r.length)};
            var output_index = ${t.indicesGet("output_indices","i")};
            var input_index = output_index * steps_i + starts_i + carry;
            carry = input_index / input_shape_i;
            input_index = input_index % input_shape_i;
            if (signs_i < 0) {
              input_index = input_shape_i - input_index - 1u + starts_i;
            }
            ${e.indicesSet("input_indices","i","input_index")};
          }
          return input_indices;
      }`,_d=(e,t)=>{let r=e[0].dims,i=B.size(r),a=t.axes.length>0?B.normalizeAxes(t.axes,r.length):[...Array(r.length).keys()],n=rr(e,4);n.forEach(v=>v!==0||(()=>{throw new Error("step cannot be 0")})),n.length===0&&(n=Array(a.length).fill(1));let s=t.starts.map((v,$)=>oa(v,$,r,a,n)),u=t.ends.map((v,$)=>oa(v,$,r,a,n));if(a.length!==s.length||a.length!==u.length)throw new Error("start, ends and axes should have the same number of elements");if(a.length!==r.length)for(let v=0;v<r.length;++v)a.includes(v)||(s.splice(v,0,0),u.splice(v,0,r[v]),n.splice(v,0,1));let d=n.map(v=>Math.sign(v));n.forEach((v,$,I)=>{if(v<0){let T=(u[$]-s[$])/v,E=s[$],A=E+T*n[$];s[$]=A,u[$]=E,I[$]=-v}});let p=r.slice(0);a.forEach((v,$)=>{p[v]=Math.ceil((u[v]-s[v])/n[v])});let c={dims:p,dataType:e[0].dataType},f=j("output",e[0].dataType,p.length),g=M("input",e[0].dataType,e[0].dims.length),_=B.size(p),w=[{name:"outputSize",type:"u32"},{name:"starts",type:"u32",length:s.length},{name:"signs",type:"i32",length:d.length},{name:"steps",type:"u32",length:n.length}],b=[{type:12,data:_},{type:12,data:s},{type:6,data:d},{type:12,data:n},...Q(e[0].dims,p)],S=v=>`
      ${v.registerUniforms(w).declareVariables(g,f)}
        ${yd(g,f,r)}
        ${v.mainStart()}
          ${v.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.outputSize")}
          let output_indices = ${f.offsetToIndices("global_idx")};
          let input_indices = calculateInputIndices(output_indices);
          ${f.setByOffset("global_idx",g.getByIndices("input_indices"))}
      }`;return{name:"Slice",shaderCache:{hint:`${d.length}_${s.length}_${n.length}`,inputDependencies:["rank"]},getShaderSource:S,getRunData:()=>({outputs:[c],dispatchGroup:{x:Math.ceil(i/64)},programUniforms:b})}},Fh=(e,t)=>{md(e.inputs,t);let r=gd(e.inputs,t);e.compute(_d(e.inputs,r),{inputs:[0]})},jh=e=>{let t=e.starts,r=e.ends,i=e.axes;return ce({starts:t,ends:r,axes:i})}}),wd,bd,Kh,Zh,M0=P(()=>{J(),ie(),ve(),yt(),ae(),wd=e=>{if(!e||e.length!==1)throw new Error("Softmax op requires 1 input.")},bd=(e,t)=>{let r=e.inputs[0],i=r.dims,a=B.size(i),n=i.length,s=B.normalizeAxis(t.axis,n),u=s<i.length-1,d,p=[];u?(p=Array.from({length:n},(C,O)=>O),p[s]=n-1,p[n-1]=s,d=e.compute(De(r,p),{inputs:[r],outputs:[-1]})[0]):d=r;let c=d.dims,f=c[n-1],g=a/f,_=$e(f),w=f/_,b=64;g===1&&(b=256);let S=(C,O)=>O===4?`max(max(${C}.x, ${C}.y), max(${C}.z, ${C}.w))`:O===2?`max(${C}.x, ${C}.y)`:O===3?`max(max(${C}.x, ${C}.y), ${C}.z)`:C,v=M("x",d.dataType,d.dims,_),$=j("result",d.dataType,d.dims,_),I=v.type.value,T=ke(d.dataType)==="f32"?`var threadMax = ${I}(-3.402823e+38f);`:`var threadMax = ${I}(-65504.0h);`,E=C=>`
      var<workgroup> rowMaxShared : ${I};
      var<workgroup> rowSumShared : ${I};
      var<workgroup> threadShared : array<${I}, ${b}>;

      fn getValue(row: i32, col: i32, row_stride: i32) -> ${I} {
        let index = row * row_stride + col;
        return x[index];
      }

      fn setValue(row: i32, col: i32, row_stride: i32, value: ${I}) {
        let index = row * row_stride + col;
        result[index] = value;
      }
      ${C.registerUniform("packedCols","i32").declareVariables(v,$)}
      ${C.mainStart(b)}
        let gindex = i32(global_idx);
        let lindex = i32(local_idx);
        const wg = ${b};
        let row = gindex / wg;
        let cols = uniforms.packedCols;
        let row_stride : i32 = uniforms.packedCols;

        // find the rows max
        ${T}
        for (var col = lindex; col < cols; col += wg) {
          let value = getValue(row, col, row_stride);
          threadMax = max(threadMax, value);
        }
        if (lindex < cols) {
          threadShared[lindex] = threadMax;
        }
        workgroupBarrier();

        var reduceSize = min(cols, wg);
        for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
          reduceSize = currSize + (reduceSize & 1);
          if (lindex < currSize) {
            threadShared[lindex] = max(threadShared[lindex], threadShared[lindex + reduceSize]);
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowMaxShared = ${I}(${S("threadShared[0]",_)});
        }
        workgroupBarrier();

        // find the rows sum
        var threadSum = ${I}(0.0);
        for (var col = lindex; col < cols; col += wg) {
          let subExp = exp(getValue(row, col, row_stride) - rowMaxShared);
          threadSum += subExp;
        }
        threadShared[lindex] = threadSum;
        workgroupBarrier();

        for (var currSize = wg >> 1;  currSize > 0; currSize = currSize >> 1) {
          if (lindex < currSize) {
            threadShared[lindex] = threadShared[lindex] + threadShared[lindex + currSize];
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowSumShared = ${I}(${gt("threadShared[0]",_)});
        }
        workgroupBarrier();

        // calculate final value for each element in the row
        for (var col = lindex; col < cols; col += wg) {
          var value = exp(getValue(row, col, row_stride) - rowMaxShared) / rowSumShared;
          // max operation protects against NaN since all values should be >=0
          value = max(value, ${I}(0.0));
          setValue(row, col, row_stride, value);
        }
      }`,A=e.compute({name:"Softmax",shaderCache:{hint:`${_};${b}`,inputDependencies:["type"]},getRunData:()=>({outputs:[{dims:c,dataType:d.dataType}],dispatchGroup:{x:g},programUniforms:[{type:6,data:w}]}),getShaderSource:E},{inputs:[d],outputs:[u?-1:0]})[0];u&&e.compute(De(A,p),{inputs:[A]})},Kh=(e,t)=>{wd(e.inputs),bd(e,t)},Zh=e=>ce({axis:e.axis})}),ua,$d,vd,xd,Qh,U0=P(()=>{J(),ie(),ae(),ua=e=>Array.from(e.getBigInt64Array(),Number),$d=e=>{if(!e||e.length!==2)throw new Error("Tile requires 2 inputs.");if(e[0].dataType!==1&&e[0].dataType!==10&&e[0].dataType!==6&&e[0].dataType!==12)throw new Error("Tile only support float, float16, int32, and uint32 data types");if(e[1].dataType!==7)throw new Error("Tile `repeats` input should be of int64 data type");if(e[1].dims.length!==1)throw new Error("Tile `repeats` input should be 1-D");if(ua(e[1]).length!==e[0].dims.length)throw new Error("Tile `repeats` input should have same number of elements as rank of input data tensor")},vd=(e,t)=>{let r=[];for(let i=0;i<e.length;++i)r.push(e[i]*t[i]);return r},xd=(e,t)=>{let r=e[0].dims,i=t??ua(e[1]),a=vd(r,i),n=B.size(a),s=e[0].dataType,u=M("input",s,r.length),d=j("output",s,a.length),p=c=>`
      const inputShape = ${u.indices(...r)};
      ${c.registerUniform("output_size","u32").declareVariables(u,d)}
      ${c.mainStart()}
      ${c.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")}
      let output_indices = ${d.offsetToIndices("global_idx")};
      var input_indices: ${u.type.indices};
      for (var i = 0; i < ${r.length}; i++) {
        let input_dim_i = ${u.indicesGet("uniforms.input_shape","i")};
        let input_dim_value = ${d.indicesGet("output_indices","i")}  % input_dim_i;

        ${u.indicesSet("input_indices","i","input_dim_value")}
      }
      ${d.setByOffset("global_idx",u.getByIndices("input_indices"))}
    }`;return{name:"Tile",shaderCache:{hint:`${i}`,inputDependencies:["rank"]},getRunData:()=>({outputs:[{dims:a,dataType:e[0].dataType}],dispatchGroup:{x:Math.ceil(n/64)},programUniforms:[{type:12,data:n},...Q(e[0].dims,a)]}),getShaderSource:p}},Qh=e=>{$d(e.inputs),e.compute(xd(e.inputs),{inputs:[0]})}}),Sd,kd,Yh,P0=P(()=>{J(),ie(),ae(),Sd=(e,t,r,i,a)=>{let n=j("output_data",a,r.length,4),s=M("a_data",t[1].dataType,t[1].dims.length,4),u=M("b_data",t[2].dataType,t[2].dims.length,4),d=M("c_data",t[0].dataType,t[0].dims.length,4),p,c=(f,g,_)=>`select(${g}, ${f}, ${_})`;if(!i)p=n.setByOffset("global_idx",c(s.getByOffset("global_idx"),u.getByOffset("global_idx"),d.getByOffset("global_idx")));else{let f=(g,_,w="")=>{let b=`a_data[index_a${_}][component_a${_}]`,S=`b_data[index_b${_}][component_b${_}]`,v=`bool(c_data[index_c${_}] & (0xffu << (component_c${_} * 8)))`;return`
            let output_indices${_} = ${n.offsetToIndices(`global_idx * 4u + ${_}u`)};
            let offset_a${_} = ${s.broadcastedIndicesToOffset(`output_indices${_}`,n)};
            let offset_b${_} = ${u.broadcastedIndicesToOffset(`output_indices${_}`,n)};
            let offset_c${_} = ${d.broadcastedIndicesToOffset(`output_indices${_}`,n)};
            let index_a${_} = offset_a${_} / 4u;
            let index_b${_} = offset_b${_} / 4u;
            let index_c${_} = offset_c${_} / 4u;
            let component_a${_} = offset_a${_} % 4u;
            let component_b${_} = offset_b${_} % 4u;
            let component_c${_} = offset_c${_} % 4u;
            ${g}[${_}] = ${w}(${c(b,S,v)});
          `};a===9?p=`
            var data = vec4<u32>(0);
            ${f("data",0,"u32")}
            ${f("data",1,"u32")}
            ${f("data",2,"u32")}
            ${f("data",3,"u32")}
            output_data[global_idx] = dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(data));`:p=`
            ${f("output_data[global_idx]",0)}
            ${f("output_data[global_idx]",1)}
            ${f("output_data[global_idx]",2)}
            ${f("output_data[global_idx]",3)}
          `}return`
        ${e.registerUniform("vec_size","u32").declareVariables(d,s,u,n)}
        ${e.mainStart()}
        ${e.guardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")}
        ${p}
      }`},kd=e=>{let t=e[1].dims,r=e[2].dims,i=e[0].dims,a=e[1].dataType,n=!(B.areEqual(t,r)&&B.areEqual(r,i)),s=t,u=B.size(t);if(n){let p=Wt.calcShape(Wt.calcShape(t,r,!1),i,!1);if(!p)throw new Error("Can't perform where op on the given tensors");s=p,u=B.size(s)}let d=Math.ceil(u/4);return{name:"Where",shaderCache:{inputDependencies:["rank","rank","rank"]},getShaderSource:p=>Sd(p,e,s,n,a),getRunData:()=>({outputs:[{dims:s,dataType:a}],dispatchGroup:{x:Math.ceil(u/64/4)},programUniforms:[{type:12,data:d},...Q(i,t,r,s)]})}},Yh=e=>{e.compute(kd(e.inputs))}}),Xh,q0=P(()=>{e0(),Va(),t0(),r0(),i0(),a0(),n0(),d0(),c0(),h0(),f0(),m0(),g0(),y0(),_0(),w0(),b0(),$0(),v0(),x0(),S0(),k0(),T0(),I0(),E0(),yh(),z0(),C0(),A0(),O0(),R0(),La(),B0(),vh(),N0(),D0(),M0(),bh(),U0(),yt(),Ga(),P0(),Xh=new Map([["Abs",[Hp]],["Acos",[Fp]],["Acosh",[jp]],["Add",[Ic]],["ArgMax",[Wp,wa]],["ArgMin",[qp,wa]],["Asin",[Kp]],["Asinh",[Zp]],["Atan",[Qp]],["Atanh",[Yp]],["Attention",[Lp]],["AveragePool",[Ah,Ch]],["BatchNormalization",[Vp]],["BiasAdd",[Gp]],["BiasSplitGelu",[Tc]],["Cast",[Jp,Xp]],["Ceil",[tc]],["Clip",[ec]],["Concat",[Mc,Uc]],["Conv",[ka,Sa]],["ConvTranspose",[Kc,jc]],["Cos",[rc]],["Cosh",[ic]],["CumSum",[Zc,Qc]],["DepthToSpace",[Yc,Xc]],["DequantizeLinear",[Uh,Ph]],["Div",[Ec]],["Einsum",[Jc,eh]],["Elu",[ac,or]],["Equal",[zc]],["Erf",[nc]],["Exp",[sc]],["Expand",[th]],["FastGelu",[rh]],["Floor",[oc]],["FusedConv",[ka,Sa]],["Gather",[ah,ih]],["GatherElements",[dh,lh]],["GatherBlockQuantized",[oh,uh]],["GatherND",[nh,sh]],["Gelu",[uc]],["Gemm",[ch,ph]],["GlobalAveragePool",[Rh,Oh]],["GlobalMaxPool",[Mh,Dh]],["Greater",[Rc]],["GreaterOrEqual",[Nc]],["GridSample",[hh,fh]],["GroupQueryAttention",[xh]],["HardSigmoid",[gc,mc]],["InstanceNormalization",[Sh]],["LayerNormalization",[kh]],["LeakyRelu",[lc,or]],["Less",[Bc]],["LessOrEqual",[Dc]],["Log",[Sc]],["MatMul",[Th]],["MatMulNBits",[Ih,Eh]],["MaxPool",[Bh,Nh]],["Mul",[Cc]],["MultiHeadAttention",[gh,mh]],["Neg",[pc]],["Not",[dc]],["Pad",[zh]],["Pow",[Ac]],["QuickGelu",[kc,or]],["Range",[qh]],["Reciprocal",[cc]],["ReduceMin",[Np]],["ReduceMean",[Cp]],["ReduceMax",[Bp]],["ReduceSum",[Mp]],["ReduceProd",[Dp]],["ReduceL1",[Ap]],["ReduceL2",[Op]],["ReduceLogSum",[Pp]],["ReduceLogSumExp",[Rp]],["ReduceSumSquare",[Up]],["Relu",[hc]],["Resize",[Vh,Gh]],["RotaryEmbedding",[$h]],["ScatterND",[Lh,Wh]],["Sigmoid",[fc]],["Sin",[yc]],["Sinh",[_c]],["Slice",[Fh,jh]],["SkipLayerNormalization",[Hh]],["Split",[_h,wh]],["Sqrt",[wc]],["Softmax",[Kh,Zh]],["Sub",[Oc]],["Tan",[bc]],["Tanh",[$c]],["ThresholdedRelu",[xc,or]],["Tile",[Qh]],["Transpose",[_p,wp]],["Where",[Yh]]])}),Jh,W0=P(()=>{Ue(),nt(),ae(),Jh=class{constructor(e){this.backend=e,this.repo=new Map,this.attributesBound=!1}getArtifact(e){return this.repo.get(e)}setArtifact(e,t){this.repo.set(e,t)}run(e,t,r,i,a){tt(e.programInfo.name);let n=this.backend.device,s=this.backend.getComputePassEncoder();this.backend.writeTimestamp(this.backend.pendingDispatchNumber*2);let u=[];for(let p of t)u.push({binding:u.length,resource:{buffer:p.buffer}});for(let p of r)u.push({binding:u.length,resource:{buffer:p.buffer}});a&&u.push({binding:u.length,resource:a});let d=n.createBindGroup({layout:e.computePipeline.getBindGroupLayout(0),entries:u,label:e.programInfo.name});if(this.backend.sessionStatus==="capturing"){let p={kernelId:this.backend.currentKernelId,computePipeline:e.computePipeline,bindGroup:d,dispatchGroup:i};this.backend.capturedCommandList.get(this.backend.currentSessionId).push(p)}s.setPipeline(e.computePipeline),s.setBindGroup(0,d),s.dispatchWorkgroups(...i),this.backend.writeTimestamp(this.backend.pendingDispatchNumber*2+1),this.backend.pendingDispatchNumber++,(this.backend.pendingDispatchNumber>=this.backend.maxDispatchNumber||this.backend.queryType==="at-passes")&&this.backend.endComputePass(),this.backend.pendingDispatchNumber>=this.backend.maxDispatchNumber&&this.backend.flush(),je(e.programInfo.name)}dispose(){}build(e,t){tt(e.name);let r=this.backend.device,i=[];[{feature:"shader-f16",extension:"f16"},{feature:"subgroups",extension:"subgroups"}].forEach(p=>{r.features.has(p.feature)&&i.push(`enable ${p.extension};`)});let a=yp(t,this.backend.device.limits),n=e.getShaderSource(a),s=`${i.join(`
`)}
${a.additionalImplementations}
${n}`,u=r.createShaderModule({code:s,label:e.name});le("verbose",()=>`[WebGPU] ${e.name} shader code: ${s}`);let d=r.createComputePipeline({compute:{module:u,entryPoint:"main"},layout:"auto",label:e.name});return je(e.name),{programInfo:e,computePipeline:d,uniformVariablesInfo:a.variablesInfo}}normalizeDispatchGroupSize(e){let t=typeof e=="number"?e:e.x,r=typeof e=="number"?1:e.y||1,i=typeof e=="number"?1:e.z||1,a=this.backend.device.limits.maxComputeWorkgroupsPerDimension;if(t<=a&&r<=a&&i<=a)return[t,r,i];let n=t*r*i,s=Math.ceil(Math.sqrt(n));if(s>a){if(s=Math.ceil(Math.cbrt(n)),s>a)throw new Error("Total dispatch size exceeds WebGPU maximum.");return[s,s,s]}else return[s,s,1]}}}),ef={};Vt(ef,{WebGpuBackend:()=>tf});var Td,Id,Ed,tf,L0=P(()=>{Ue(),J(),nt(),cp(),Xg(),q0(),W0(),Td=(e,t)=>{if(t.length!==e.length)throw new Error(`inputDependencies length ${t.length} is not equal to inputTensors length ${e.length}.`);let r=[];for(let i=0;i<e.length;++i){let a=e[i].dataType;switch(t[i]){case"none":{r.push("");break}case"type":{r.push(`${a}`);break}case"rank":{let n=e[i].dims.length;r.push(`${a};${n}`);break}case"dims":{let n=e[i].dims.join(",");r.push(`${a};${n}`);break}default:throw new Error(`unsupported input dependency: ${t[i]}`)}}return r.join("|")},Id=(e,t,r)=>{var a,n;let i=e.name;return(a=e.shaderCache)!=null&&a.hint&&(i+="["+e.shaderCache.hint+"]"),i+=":"+r+`:${Td(t,((n=e.shaderCache)==null?void 0:n.inputDependencies)??new Array(t.length).fill("dims"))}`,i},Ed=class{constructor(e){e&&(this.architecture=e.architecture,this.vendor=e.vendor)}isArchitecture(e){return this.architecture===e}isVendor(e){return this.vendor===e}},tf=class{constructor(){this.currentSessionId=null,this.currentKernelId=null,this.commandEncoder=null,this.computePassEncoder=null,this.maxDispatchNumber=16,this.pendingDispatchNumber=0,this.pendingKernels=[],this.pendingQueries=new Map,this.sessionStatus="default",this.capturedCommandList=new Map,this.capturedPendingKernels=new Map,this.sessionExternalDataMapping=new Map}get currentKernelCustomData(){if(this.currentKernelId===null)throw new Error("currentKernelCustomData(): currentKernelId is null. (should not happen)");let e=this.kernelCustomData.get(this.currentKernelId);return e||(e={},this.kernelCustomData.set(this.currentKernelId,e)),e}async initialize(e,t){this.env=e;let r=[],i={requiredLimits:{maxComputeWorkgroupStorageSize:t.limits.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:t.limits.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:t.limits.maxStorageBufferBindingSize,maxBufferSize:t.limits.maxBufferSize,maxComputeInvocationsPerWorkgroup:t.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupSizeX:t.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupSizeY:t.limits.maxComputeWorkgroupSizeY,maxComputeWorkgroupSizeZ:t.limits.maxComputeWorkgroupSizeZ},requiredFeatures:r},a=n=>t.features.has(n)&&r.push(n)&&!0;a("chromium-experimental-timestamp-query-inside-passes")||a("timestamp-query"),a("shader-f16"),a("subgroups"),this.device=await t.requestDevice(i),this.adapterInfo=new Ed(t.info||await t.requestAdapterInfo()),this.gpuDataManager=mp(this),this.programManager=new Jh(this),this.kernels=new Map,this.kernelPersistentData=new Map,this.kernelCustomData=new Map,Ua(e.logLevel,!!e.debug),this.device.onuncapturederror=n=>{n.error instanceof GPUValidationError&&console.error(`An uncaught WebGPU validation error was raised: ${n.error.message}`)},Object.defineProperty(this.env.webgpu,"device",{value:this.device,writable:!1,enumerable:!0,configurable:!1}),Object.defineProperty(this.env.webgpu,"adapter",{value:t,writable:!1,enumerable:!0,configurable:!1}),this.setQueryType()}dispose(){typeof this.querySet<"u"&&this.querySet.destroy(),this.gpuDataManager.dispose()}getCommandEncoder(){return this.commandEncoder||(this.commandEncoder=this.device.createCommandEncoder()),this.commandEncoder}getComputePassEncoder(){if(!this.computePassEncoder){let e=this.getCommandEncoder(),t={};this.queryType==="at-passes"&&(t.timestampWrites={querySet:this.querySet,beginningOfPassWriteIndex:this.pendingDispatchNumber*2,endOfPassWriteIndex:this.pendingDispatchNumber*2+1}),this.computePassEncoder=e.beginComputePass(t)}return this.computePassEncoder}endComputePass(){this.computePassEncoder&&(this.computePassEncoder.end(),this.computePassEncoder=null)}flush(){if(!this.commandEncoder)return;tt(),this.endComputePass();let e;this.queryType!=="none"&&(this.commandEncoder.resolveQuerySet(this.querySet,0,this.pendingDispatchNumber*2,this.queryResolveBuffer,0),e=this.device.createBuffer({size:this.pendingDispatchNumber*2*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),this.pendingQueries.set(e,this.pendingKernels),this.pendingKernels=[],this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer,0,e,0,this.pendingDispatchNumber*2*8)),this.device.queue.submit([this.commandEncoder.finish()]),this.gpuDataManager.refreshPendingBuffers(),this.commandEncoder=null,this.pendingDispatchNumber=0,this.queryType!=="none"&&e.mapAsync(GPUMapMode.READ).then(()=>{var i;let t=new BigUint64Array(e.getMappedRange()),r=this.pendingQueries.get(e);for(let a=0;a<t.length/2;a++){let n=r[a],s=n.kernelId,u=this.kernels.get(s),d=u.kernelType,p=u.kernelName,c=n.programName,f=n.inputTensorViews,g=n.outputTensorViews,_=t[a*2],w=t[a*2+1];typeof this.queryTimeBase>"u"&&(this.queryTimeBase=_);let b=Number(_-this.queryTimeBase),S=Number(w-this.queryTimeBase);if(!Number.isSafeInteger(b)||!Number.isSafeInteger(S))throw new RangeError("incorrect timestamp range");if((i=this.env.webgpu.profiling)!=null&&i.ondata)this.env.webgpu.profiling.ondata({version:1,inputsMetadata:f.map(v=>({dims:v.dims,dataType:at(v.dataType)})),outputsMetadata:g.map(v=>({dims:v.dims,dataType:at(v.dataType)})),kernelId:s,kernelType:d,kernelName:p,programName:c,startTime:b,endTime:S});else{let v="";f.forEach((I,T)=>{v+=`input[${T}]: [${I.dims}] | ${at(I.dataType)}, `});let $="";g.forEach((I,T)=>{$+=`output[${T}]: [${I.dims}] | ${at(I.dataType)}, `}),console.log(`[profiling] kernel "${s}|${d}|${p}|${c}" ${v}${$}start time: ${b} ns, execution time: ${S-b} ns`)}Ur("GPU",`${c}::${_}::${w}`)}e.unmap(),this.pendingQueries.delete(e)}),je()}run(e,t,r,i,a,n){tt(e.name);let s=[];for(let $=0;$<t.length;++$){let I=t[$].data;if(I===0)continue;let T=this.gpuDataManager.get(I);if(!T)throw new Error(`no GPU data for input: ${I}`);s.push(T)}let{outputs:u,dispatchGroup:d,programUniforms:p}=e.getRunData(t),c=r.length===0?u.map(($,I)=>I):r;if(c.length!==u.length)throw new Error(`Output size ${c.length} must be equal to ${u.length}.`);let f=[],g=[];for(let $=0;$<u.length;++$){if(!Number.isInteger(c[$])||c[$]<-3||c[$]>=n)throw new Error(`Invalid output index: ${c[$]}`);if(c[$]===-3)continue;let I=c[$]===-1,T=c[$]===-2,E=I||T?a(u[$].dataType,u[$].dims):i(c[$],u[$].dataType,u[$].dims);if(f.push(E),E.data===0)continue;let A=this.gpuDataManager.get(E.data);if(!A)throw new Error(`no GPU data for output: ${E.data}`);if(I&&this.temporaryData.push(A),T){let C=this.kernelPersistentData.get(this.currentKernelId);C||(C=[],this.kernelPersistentData.set(this.currentKernelId,C)),C.push(A)}g.push(A)}if(s.length!==t.length||g.length!==f.length){if(g.length===0)return je(e.name),f;throw new Error(`Program ${e.name} has zero-sized tensor(s) in inputs or outputs. This is not supported now.`)}let _;if(p){let $=0,I=[];p.forEach(C=>{let O=typeof C.data=="number"?[C.data]:C.data;if(O.length===0)return;let U=C.type===10?2:4,x,Y;C.type===10?(Y=O.length>4?16:O.length>2?8:O.length*U,x=O.length>4?16:U*O.length):(Y=O.length<=2?O.length*U:16,x=16),$=Math.ceil($/Y)*Y,I.push($);let G=C.type===10?8:4;$+=O.length>4?Math.ceil(O.length/G)*x:O.length*U});let T=16;$=Math.ceil($/T)*T;let E=new ArrayBuffer($);p.forEach((C,O)=>{let U=I[O],x=typeof C.data=="number"?[C.data]:C.data;if(C.type===6)new Int32Array(E,U,x.length).set(x);else if(C.type===12)new Uint32Array(E,U,x.length).set(x);else if(C.type===10)new Uint16Array(E,U,x.length).set(x);else if(C.type===1)new Float32Array(E,U,x.length).set(x);else throw new Error(`Unsupported uniform type: ${at(C.type)}`)});let A=this.gpuDataManager.create($,GPUBufferUsage.COPY_DST|GPUBufferUsage.UNIFORM);this.device.queue.writeBuffer(A.buffer,0,E,0,$),this.gpuDataManager.release(A.id),_={offset:0,size:$,buffer:A.buffer}}let w=this.programManager.normalizeDispatchGroupSize(d),b=w[1]===1&&w[2]===1,S=Id(e,t,b),v=this.programManager.getArtifact(S);if(v||(v=this.programManager.build(e,w),this.programManager.setArtifact(S,v),le("info",()=>`[artifact] key: ${S}, programName: ${e.name}`)),p&&v.uniformVariablesInfo){if(p.length!==v.uniformVariablesInfo.length)throw new Error(`Uniform variables count mismatch: expect ${v.uniformVariablesInfo.length}, got ${p.length} in program "${v.programInfo.name}".`);for(let $=0;$<p.length;$++){let I=p[$],T=I.type,E=typeof I.data=="number"?1:I.data.length,[A,C]=v.uniformVariablesInfo[$];if(T!==A||E!==C)throw new Error(`Uniform variable ${$} mismatch: expect type ${A} with size ${C}, got type ${T} with size ${E} in program "${v.programInfo.name}".`)}}if(le("info",()=>`[ProgramManager] run "${e.name}" (key=${S}) with ${w[0]}x${w[1]}x${w[2]}`),this.queryType!=="none"||this.sessionStatus==="capturing"){let $={kernelId:this.currentKernelId,programName:v.programInfo.name,inputTensorViews:t,outputTensorViews:f};this.pendingKernels.push($),this.sessionStatus==="capturing"&&this.capturedPendingKernels.get(this.currentSessionId).push($)}return this.programManager.run(v,s,g,w,_),je(e.name),f}upload(e,t){this.gpuDataManager.upload(e,t)}memcpy(e,t){this.gpuDataManager.memcpy(e,t)}async download(e,t){await this.gpuDataManager.download(e,t)}alloc(e){return this.gpuDataManager.create(e).id}free(e){return this.gpuDataManager.release(e)}createKernel(e,t,r,i){let a=Xh.get(e);if(!a)throw new Error(`kernel not implemented: ${e}`);let n={kernelType:e,kernelName:i,kernelEntry:a[0],attributes:[a[1],r]};this.kernels.set(t,n)}releaseKernel(e){let t=this.kernelPersistentData.get(e);if(t){for(let r of t)this.gpuDataManager.release(r.id);this.kernelPersistentData.delete(e)}this.kernelCustomData.delete(e),this.kernels.delete(e)}computeKernel(e,t,r){let i=this.kernels.get(e);if(!i)throw new Error(`kernel not created: ${e}`);let a=i.kernelType,n=i.kernelName,s=i.kernelEntry,u=i.attributes;if(this.currentKernelId!==null)throw new Error(`kernel "[${a}] ${n}" is not allowed to be called recursively`);this.currentKernelId=e,u[0]&&(u[1]=u[0](u[1]),u[0]=void 0),le("info",()=>`[WebGPU] Start to run kernel "[${a}] ${n}"...`);let d=this.env.debug;this.temporaryData=[];try{return d&&this.device.pushErrorScope("validation"),s(t,u[1]),0}catch(p){return r.push(Promise.resolve(`[WebGPU] Kernel "[${a}] ${n}" failed. ${p}`)),1}finally{d&&r.push(this.device.popErrorScope().then(p=>p?`GPU validation error for kernel "[${a}] ${n}": ${p.message}`:null));for(let p of this.temporaryData)this.gpuDataManager.release(p.id);this.temporaryData=[],this.currentKernelId=null}}registerBuffer(e,t,r,i){let a=this.sessionExternalDataMapping.get(e);a||(a=new Map,this.sessionExternalDataMapping.set(e,a));let n=a.get(t),s=this.gpuDataManager.registerExternalBuffer(r,i,n);return a.set(t,[s,r]),s}unregisterBuffers(e){let t=this.sessionExternalDataMapping.get(e);t&&(t.forEach(r=>this.gpuDataManager.unregisterExternalBuffer(r[0])),this.sessionExternalDataMapping.delete(e))}getBuffer(e){let t=this.gpuDataManager.get(e);if(!t)throw new Error(`no GPU data for buffer: ${e}`);return t.buffer}createDownloader(e,t,r){return async()=>{let i=await ga(this,e,t);return Pa(i.buffer,r)}}writeTimestamp(e){this.queryType==="inside-passes"&&this.computePassEncoder.writeTimestamp(this.querySet,e)}setQueryType(){var e;this.queryType="none",(((e=this.env.webgpu.profiling)==null?void 0:e.mode)==="default"||(typeof this.env.trace>"u"?this.env.wasm.trace:this.env.trace))&&(this.device.features.has("chromium-experimental-timestamp-query-inside-passes")?this.queryType="inside-passes":this.device.features.has("timestamp-query")&&(this.queryType="at-passes"),this.queryType!=="none"&&typeof this.querySet>"u"&&(this.querySet=this.device.createQuerySet({type:"timestamp",count:this.maxDispatchNumber*2}),this.queryResolveBuffer=this.device.createBuffer({size:this.maxDispatchNumber*2*8,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.QUERY_RESOLVE})))}captureBegin(){le("info","captureBegin"),this.capturedCommandList.get(this.currentSessionId)||this.capturedCommandList.set(this.currentSessionId,[]),this.capturedPendingKernels.get(this.currentSessionId)||this.capturedPendingKernels.set(this.currentSessionId,[]),this.flush(),this.sessionStatus="capturing"}captureEnd(){le("info","captureEnd"),this.flush(),this.sessionStatus="default"}replay(){le("info","replay"),this.sessionStatus="replaying";let e=this.capturedCommandList.get(this.currentSessionId),t=this.capturedPendingKernels.get(this.currentSessionId),r=e.length;this.pendingKernels=[];for(let i=0;i<r;i++){let a=this.getComputePassEncoder(),n=e[i];this.writeTimestamp(this.pendingDispatchNumber*2),a.setPipeline(n.computePipeline),a.setBindGroup(0,n.bindGroup),a.dispatchWorkgroups(...n.dispatchGroup),this.writeTimestamp(this.pendingDispatchNumber*2+1),this.pendingDispatchNumber++,this.queryType!=="none"&&this.pendingKernels.push(t[i]),(this.pendingDispatchNumber>=this.maxDispatchNumber||this.queryType==="at-passes")&&this.endComputePass(),this.pendingDispatchNumber>=this.maxDispatchNumber&&this.flush()}this.flush(),this.sessionStatus="default"}onCreateSession(){this.gpuDataManager.onCreateSession()}onReleaseSession(e){this.unregisterBuffers(e),this.capturedCommandList.has(e)&&this.capturedCommandList.delete(e),this.capturedPendingKernels.has(e)&&this.capturedPendingKernels.delete(e),this.gpuDataManager.onReleaseSession(e)}onRunStart(e){this.currentSessionId=e,this.setQueryType()}}}),rf={};Vt(rf,{init:()=>af});var Rr,zd,af,V0=P(()=>{J(),nt(),ie(),Yg(),Rr=class nf{constructor(t,r,i,a){this.module=t,this.dataType=r,this.data=i,this.dims=a}getFloat32Array(){if(this.dataType!==1)throw new Error("Invalid data type");let t=B.size(this.dims);return t===0?new Float32Array:new Float32Array(this.module.HEAP8.buffer,this.data,t)}getBigInt64Array(){if(this.dataType!==7)throw new Error("Invalid data type");let t=B.size(this.dims);return t===0?new BigInt64Array:new BigInt64Array(this.module.HEAP8.buffer,this.data,t)}getInt32Array(){if(this.dataType!==6)throw new Error("Invalid data type");let t=B.size(this.dims);return t===0?new Int32Array:new Int32Array(this.module.HEAP8.buffer,this.data,t)}getUint16Array(){if(this.dataType!==10&&this.dataType!==4)throw new Error("Invalid data type");let t=B.size(this.dims);return t===0?new Uint16Array:new Uint16Array(this.module.HEAP8.buffer,this.data,t)}reshape(t){if(B.size(t)!==B.size(this.dims))throw new Error("Invalid new shape");return new nf(this.module,this.dataType,this.data,t)}},zd=class{constructor(e,t,r){this.module=e,this.backend=t,this.customDataOffset=0,this.customDataSize=0,this.adapterInfo=t.adapterInfo;let i=e.PTR_SIZE,a=r/e.PTR_SIZE,n=i===4?"i32":"i64";this.opKernelContext=Number(e.getValue(i*a++,n));let s=Number(e.getValue(i*a++,n));this.outputCount=Number(e.getValue(i*a++,n)),this.customDataOffset=Number(e.getValue(i*a++,"*")),this.customDataSize=Number(e.getValue(i*a++,n));let u=[];for(let d=0;d<s;d++){let p=Number(e.getValue(i*a++,n)),c=Number(e.getValue(i*a++,"*")),f=Number(e.getValue(i*a++,n)),g=[];for(let _=0;_<f;_++)g.push(Number(e.getValue(i*a++,n)));u.push(new Rr(e,p,c,g))}this.inputs=u}get kernelCustomData(){return this.backend.currentKernelCustomData}get customDataBuffer(){return this.module.HEAPU8.subarray(this.customDataOffset,this.customDataOffset+this.customDataSize)}compute(e,t){var s;let r=((s=t==null?void 0:t.inputs)==null?void 0:s.map(u=>typeof u=="number"?this.inputs[u]:u))??this.inputs,i=(t==null?void 0:t.outputs)??[],a=(u,d,p)=>new Rr(this.module,d,this.output(u,p),p),n=(u,d)=>{let p=Et(u,d);if(!p)throw new Error(`Unsupported data type: ${u}`);let c=p>0?this.backend.gpuDataManager.create(p).id:0;return new Rr(this.module,u,c,d)};return this.backend.run(e,r,i,a,n,this.outputCount)}output(e,t){let r=this.module.stackSave();try{let i=this.module.PTR_SIZE,a=i===4?"i32":"i64",n=this.module.stackAlloc((1+t.length)*i);this.module.setValue(n,t.length,a);for(let s=0;s<t.length;s++)this.module.setValue(n+i*(s+1),t[s],a);return this.module._JsepOutput(this.opKernelContext,e,n)}catch(i){throw new Error(`Failed to generate kernel's output[${e}] with dims [${t}]. If you are running with pre-allocated output, please make sure the output type/dims are correct. Error: ${i}`)}finally{this.module.stackRestore(r)}}},af=async(e,t,r,i)=>{let a=t.jsepInit;if(!a)throw new Error("Failed to initialize JSEP. The WebAssembly module is not built with JSEP support.");if(e==="webgpu"){let n=(L0(),dr(ef)).WebGpuBackend,s=new n;await s.initialize(r,i),a("webgpu",[s,u=>s.alloc(Number(u)),u=>s.free(u),(u,d,p,c=!1)=>{if(c)le("verbose",()=>`[WebGPU] jsepCopyGpuToGpu: src=${Number(u)}, dst=${Number(d)}, size=${Number(p)}`),s.memcpy(Number(u),Number(d));else{le("verbose",()=>`[WebGPU] jsepCopyCpuToGpu: dataOffset=${Number(u)}, gpuDataId=${Number(d)}, size=${Number(p)}`);let f=t.HEAPU8.subarray(Number(u>>>0),Number(u>>>0)+Number(p));s.upload(Number(d),f)}},async(u,d,p)=>{le("verbose",()=>`[WebGPU] jsepCopyGpuToCpu: gpuDataId=${u}, dataOffset=${d}, size=${p}`),await s.download(Number(u),()=>t.HEAPU8.subarray(Number(d)>>>0,Number(d+p)>>>0))},(u,d,p)=>s.createKernel(u,Number(d),p,t.UTF8ToString(t._JsepGetNodeName(Number(d)))),u=>s.releaseKernel(u),(u,d,p,c)=>{le("verbose",()=>`[WebGPU] jsepRun: sessionHandle=${p}, kernel=${u}, contextDataOffset=${d}`);let f=new zd(t,s,Number(d));return s.computeKernel(Number(u),f,c)},()=>s.captureBegin(),()=>s.captureEnd(),()=>s.replay()])}else{let n=new fp(r);a("webnn",[n,()=>n.reserveTensorId(),s=>n.releaseTensorId(s),async(s,u,d,p,c)=>n.ensureTensor(s,u,d,p,c),(s,u)=>{n.uploadTensor(s,u)},async(s,u)=>n.downloadTensor(s,u),(s,u)=>n.registerMLContext(s,u),!!r.trace])}}}),Cd,Qa,Ya,ft,Ad,la,Hr,Xa,Ja,da,en,tn,rn,sf=P(()=>{Ue(),Kg(),Zg(),J(),Bt(),Ba(),up(),Cd=(e,t)=>{me()._OrtInit(e,t)!==0&&he("Can't initialize onnxruntime.")},Qa=async e=>{Cd(e.wasm.numThreads,qr(e.logLevel))},Ya=async(e,t)=>{var i,a;(a=(i=me()).asyncInit)==null||a.call(i);let r=e.webgpu.adapter;if(t==="webgpu"){if(typeof navigator>"u"||!navigator.gpu)throw new Error("WebGPU is not supported in current environment");if(r){if(typeof r.limits!="object"||typeof r.features!="object"||typeof r.requestDevice!="function")throw new Error("Invalid GPU adapter set in `env.webgpu.adapter`. It must be a GPUAdapter object.")}else{let n=e.webgpu.powerPreference;if(n!==void 0&&n!=="low-power"&&n!=="high-performance")throw new Error(`Invalid powerPreference setting: "${n}"`);let s=e.webgpu.forceFallbackAdapter;if(s!==void 0&&typeof s!="boolean")throw new Error(`Invalid forceFallbackAdapter setting: "${s}"`);if(r=await navigator.gpu.requestAdapter({powerPreference:n,forceFallbackAdapter:s}),!r)throw new Error('Failed to get GPU adapter. You may need to enable flag "--enable-unsafe-webgpu" if you are using Chrome.')}}if(t==="webnn"&&(typeof navigator>"u"||!navigator.ml))throw new Error("WebNN is not supported in current environment");{let n=(V0(),dr(rf)).init;t==="webgpu"&&await n("webgpu",me(),e,r),t==="webnn"&&await n("webnn",me(),e)}},ft=new Map,Ad=e=>{let t=me(),r=t.stackSave();try{let i=t.PTR_SIZE,a=t.stackAlloc(2*i);t._OrtGetInputOutputCount(e,a,a+i)!==0&&he("Can't get session input/output count.");let n=i===4?"i32":"i64";return[Number(t.getValue(a,n)),Number(t.getValue(a+i,n))]}finally{t.stackRestore(r)}},la=(e,t)=>{let r=me(),i=r.stackSave(),a=0;try{let n=r.PTR_SIZE,s=r.stackAlloc(2*n);r._OrtGetInputOutputMetadata(e,t,s,s+n)!==0&&he("Can't get session input/output metadata.");let u=Number(r.getValue(s,"*"));a=Number(r.getValue(s+n,"*"));let d=r.HEAP32[a/4];if(d===0)return[u,0];let p=r.HEAPU32[a/4+1],c=[];for(let f=0;f<p;f++){let g=Number(r.getValue(a+8+f*n,"*"));c.push(g!==0?r.UTF8ToString(g):Number(r.getValue(a+8+(f+p)*n,"*")))}return[u,d,c]}finally{r.stackRestore(i),a!==0&&r._OrtFree(a)}},Hr=e=>{let t=me(),r=t._malloc(e.byteLength);if(r===0)throw new Error(`Can't create a session. failed to allocate a buffer of size ${e.byteLength}.`);return t.HEAPU8.set(e,r),[r,e.byteLength]},Xa=async(e,t)=>{var f,g,_,w;let r,i,a=me();Array.isArray(e)?[r,i]=e:e.buffer===a.HEAPU8.buffer?[r,i]=[e.byteOffset,e.byteLength]:[r,i]=Hr(e);let n=0,s=0,u=0,d=[],p=[],c=[];try{if([s,d]=await op(t),(t==null?void 0:t.externalData)&&a.mountExternalData){let O=[];for(let U of t.externalData){let x=typeof U=="string"?U:U.path;O.push(Ma(typeof U=="string"?U:U.data).then(Y=>{a.mountExternalData(x,Y)}))}await Promise.all(O)}for(let O of(t==null?void 0:t.executionProviders)??[])if((typeof O=="string"?O:O.name)==="webnn"){if(a.shouldTransferToMLTensor=!1,typeof O!="string"){let U=O,x=U==null?void 0:U.context,Y=U==null?void 0:U.gpuDevice,G=U==null?void 0:U.deviceType,V=U==null?void 0:U.powerPreference;x?a.currentContext=x:Y?a.currentContext=await a.webnnCreateMLContext(Y):a.currentContext=await a.webnnCreateMLContext({deviceType:G,powerPreference:V})}else a.currentContext=await a.webnnCreateMLContext();break}n=await a._OrtCreateSession(r,i,s),(f=a.webgpuOnCreateSession)==null||f.call(a,n),n===0&&he("Can't create a session."),(g=a.jsepOnCreateSession)==null||g.call(a),a.currentContext&&(a.webnnRegisterMLContext(n,a.currentContext),a.currentContext=void 0,a.shouldTransferToMLTensor=!0);let[b,S]=Ad(n),v=!!(t!=null&&t.enableGraphCapture),$=[],I=[],T=[],E=[],A=[];for(let O=0;O<b;O++){let[U,x,Y]=la(n,O);U===0&&he("Can't get an input name."),p.push(U);let G=a.UTF8ToString(U);$.push(G),T.push(x===0?{name:G,isTensor:!1}:{name:G,isTensor:!0,type:at(x),shape:Y})}for(let O=0;O<S;O++){let[U,x,Y]=la(n,O+b);U===0&&he("Can't get an output name."),c.push(U);let G=a.UTF8ToString(U);I.push(G),E.push(x===0?{name:G,isTensor:!1}:{name:G,isTensor:!0,type:at(x),shape:Y});{if(v&&(t==null?void 0:t.preferredOutputLocation)===void 0){A.push("gpu-buffer");continue}let V=typeof(t==null?void 0:t.preferredOutputLocation)=="string"?t.preferredOutputLocation:((_=t==null?void 0:t.preferredOutputLocation)==null?void 0:_[G])??"cpu",te=a.webnnIsGraphOutput;if(V==="cpu"&&te&&te(n,G)){A.push("ml-tensor-cpu-output");continue}if(V!=="cpu"&&V!=="cpu-pinned"&&V!=="gpu-buffer"&&V!=="ml-tensor")throw new Error(`Not supported preferred output location: ${V}.`);if(v&&V!=="gpu-buffer")throw new Error(`Not supported preferred output location: ${V}. Only 'gpu-buffer' location is supported when enableGraphCapture is true.`);A.push(V)}}let C=null;return A.some(O=>O==="gpu-buffer"||O==="ml-tensor"||O==="ml-tensor-cpu-output")&&(u=a._OrtCreateBinding(n),u===0&&he("Can't create IO binding."),C={handle:u,outputPreferredLocations:A,outputPreferredLocationsEncoded:A.map(O=>O==="ml-tensor-cpu-output"?"ml-tensor":O).map(O=>fa(O))}),ft.set(n,[n,p,c,C,v,!1]),[n,$,I,T,E]}catch(b){throw p.forEach(S=>a._OrtFree(S)),c.forEach(S=>a._OrtFree(S)),u!==0&&a._OrtReleaseBinding(u)!==0&&he("Can't release IO binding."),n!==0&&a._OrtReleaseSession(n)!==0&&he("Can't release session."),b}finally{a._free(r),s!==0&&a._OrtReleaseSessionOptions(s)!==0&&he("Can't release session options."),d.forEach(b=>a._free(b)),(w=a.unmountExternalData)==null||w.call(a)}},Ja=e=>{var d,p,c;let t=me(),r=ft.get(e);if(!r)throw new Error(`cannot release session. invalid session id: ${e}`);let[i,a,n,s,u]=r;s&&(u&&t._OrtClearBoundOutputs(s.handle)!==0&&he("Can't clear bound outputs."),t._OrtReleaseBinding(s.handle)!==0&&he("Can't release IO binding.")),(d=t.jsepOnReleaseSession)==null||d.call(t,e),(p=t.webnnOnReleaseSession)==null||p.call(t,e),(c=t.webgpuOnReleaseSession)==null||c.call(t,e),a.forEach(f=>t._OrtFree(f)),n.forEach(f=>t._OrtFree(f)),t._OrtReleaseSession(i)!==0&&he("Can't release session."),ft.delete(e)},da=async(e,t,r,i,a,n,s=!1)=>{if(!e){t.push(0);return}let u=me(),d=u.PTR_SIZE,p=e[0],c=e[1],f=e[3],g=f,_,w;if(p==="string"&&(f==="gpu-buffer"||f==="ml-tensor"))throw new Error("String tensor is not supported on GPU.");if(s&&f!=="gpu-buffer")throw new Error(`External buffer must be provided for input/output index ${n} when enableGraphCapture is true.`);if(f==="gpu-buffer"){let v=e[2].gpuBuffer;w=Et(It(p),c);{let $=u.jsepRegisterBuffer;if(!$)throw new Error('Tensor location "gpu-buffer" is not supported without using WebGPU.');_=$(i,n,v,w)}}else if(f==="ml-tensor"){let v=e[2].mlTensor;w=Et(It(p),c);let $=u.webnnRegisterMLTensor;if(!$)throw new Error('Tensor location "ml-tensor" is not supported without using WebNN.');_=$(i,v,It(p),c)}else{let v=e[2];if(Array.isArray(v)){w=d*v.length,_=u._malloc(w),r.push(_);for(let $=0;$<v.length;$++){if(typeof v[$]!="string")throw new TypeError(`tensor data at index ${$} is not a string`);u.setValue(_+$*d,Fe(v[$],r),"*")}}else{let $=u.webnnIsGraphInput,I=u.webnnIsGraphOutput;if(p!=="string"&&$&&I){let T=u.UTF8ToString(a);if($(i,T)||I(i,T)){let E=It(p);w=Et(E,c),g="ml-tensor";let A=u.webnnCreateTemporaryTensor,C=u.webnnUploadTensor;if(!A||!C)throw new Error('Tensor location "ml-tensor" is not supported without using WebNN.');let O=await A(i,E,c);C(O,new Uint8Array(v.buffer,v.byteOffset,v.byteLength)),_=O}else w=v.byteLength,_=u._malloc(w),r.push(_),u.HEAPU8.set(new Uint8Array(v.buffer,v.byteOffset,w),_)}else w=v.byteLength,_=u._malloc(w),r.push(_),u.HEAPU8.set(new Uint8Array(v.buffer,v.byteOffset,w),_)}}let b=u.stackSave(),S=u.stackAlloc(4*c.length);try{c.forEach(($,I)=>u.setValue(S+I*d,$,d===4?"i32":"i64"));let v=u._OrtCreateTensor(It(p),_,w,S,c.length,fa(g));v===0&&he(`Can't create tensor for input/output. session=${i}, index=${n}.`),t.push(v)}finally{u.stackRestore(b)}},en=async(e,t,r,i,a,n)=>{var Y,G,V,te;let s=me(),u=s.PTR_SIZE,d=ft.get(e);if(!d)throw new Error(`cannot run inference. invalid session id: ${e}`);let p=d[0],c=d[1],f=d[2],g=d[3],_=d[4],w=d[5],b=t.length,S=i.length,v=0,$=[],I=[],T=[],E=[],A=s.stackSave(),C=s.stackAlloc(b*u),O=s.stackAlloc(b*u),U=s.stackAlloc(S*u),x=s.stackAlloc(S*u);try{[v,$]=sp(n),zt("wasm prepareInputOutputTensor");for(let q=0;q<b;q++)await da(r[q],I,E,e,c[t[q]],t[q],_);for(let q=0;q<S;q++)await da(a[q],T,E,e,f[i[q]],b+i[q],_);Ct("wasm prepareInputOutputTensor");for(let q=0;q<b;q++)s.setValue(C+q*u,I[q],"*"),s.setValue(O+q*u,c[t[q]],"*");for(let q=0;q<S;q++)s.setValue(U+q*u,T[q],"*"),s.setValue(x+q*u,f[i[q]],"*");if(g&&!w){let{handle:q,outputPreferredLocations:X,outputPreferredLocationsEncoded:_e}=g;if(c.length!==b)throw new Error(`input count from feeds (${b}) is expected to be always equal to model's input count (${c.length}).`);zt("wasm bindInputsOutputs");for(let D=0;D<b;D++){let L=t[D];await s._OrtBindInput(q,c[L],I[D])!==0&&he(`Can't bind input[${D}] for session=${e}.`)}for(let D=0;D<S;D++){let L=i[D];(Y=a[D])!=null&&Y[3]?s._OrtBindOutput(q,f[L],T[D],0)!==0&&he(`Can't bind pre-allocated output[${D}] for session=${e}.`):s._OrtBindOutput(q,f[L],0,_e[L])!==0&&he(`Can't bind output[${D}] to ${X[D]} for session=${e}.`)}Ct("wasm bindInputsOutputs"),ft.set(e,[p,c,f,g,_,!0])}(G=s.jsepOnRunStart)==null||G.call(s,p),(V=s.webnnOnRunStart)==null||V.call(s,p);let ee;g?ee=await s._OrtRunWithBinding(p,g.handle,S,U,v):ee=await s._OrtRun(p,O,C,b,x,S,U,v),ee!==0&&he("failed to call OrtRun().");let F=[],R=[];zt("wasm ProcessOutputTensor");for(let q=0;q<S;q++){let X=Number(s.getValue(U+q*u,"*"));if(X===T[q]){F.push(a[q]);continue}let _e=s.stackSave(),D=s.stackAlloc(4*u),L=!1,K,re=0;try{s._OrtGetTensorData(X,D,D+u,D+2*u,D+3*u)!==0&&he(`Can't access output tensor data on index ${q}.`);let ze=u===4?"i32":"i64",Ke=Number(s.getValue(D,ze));re=s.getValue(D+u,"*");let st=s.getValue(D+u*2,"*"),_t=Number(s.getValue(D+u*3,ze)),Pe=[];for(let xe=0;xe<_t;xe++)Pe.push(Number(s.getValue(st+xe*u,ze)));s._OrtFree(st)!==0&&he("Can't free memory for tensor dims.");let Te=Pe.reduce((xe,be)=>xe*be,1);K=at(Ke);let ot=g==null?void 0:g.outputPreferredLocations[i[q]];if(K==="string"){if(ot==="gpu-buffer"||ot==="ml-tensor")throw new Error("String tensor is not supported on GPU.");let xe=[];for(let be=0;be<Te;be++){let Re=s.getValue(re+be*u,"*"),cr=s.getValue(re+(be+1)*u,"*"),Ze=be===Te-1?void 0:cr-Re;xe.push(s.UTF8ToString(Re,Ze))}F.push([K,Pe,xe,"cpu"])}else if(ot==="gpu-buffer"&&Te>0){let xe=s.jsepGetBuffer;if(!xe)throw new Error('preferredLocation "gpu-buffer" is not supported without using WebGPU.');let be=xe(re),Re=Et(Ke,Te);if(Re===void 0||!Na(K))throw new Error(`Unsupported data type: ${K}`);L=!0,F.push([K,Pe,{gpuBuffer:be,download:s.jsepCreateDownloader(be,Re,K),dispose:()=>{s._OrtReleaseTensor(X)!==0&&he("Can't release tensor.")}},"gpu-buffer"])}else if(ot==="ml-tensor"&&Te>0){let xe=s.webnnEnsureTensor,be=s.webnnIsGraphInputOutputTypeSupported;if(!xe||!be)throw new Error('preferredLocation "ml-tensor" is not supported without using WebNN.');if(Et(Ke,Te)===void 0||!Da(K))throw new Error(`Unsupported data type: ${K}`);if(!be(e,K,!1))throw new Error(`preferredLocation "ml-tensor" for ${K} output is not supported by current WebNN Context.`);let Re=await xe(e,re,Ke,Pe,!1);L=!0,F.push([K,Pe,{mlTensor:Re,download:s.webnnCreateMLTensorDownloader(re,K),dispose:()=>{s.webnnReleaseTensorId(re),s._OrtReleaseTensor(X)}},"ml-tensor"])}else if(ot==="ml-tensor-cpu-output"&&Te>0){let xe=s.webnnCreateMLTensorDownloader(re,K)(),be=F.length;L=!0,R.push((async()=>{let Re=[be,await xe];return s.webnnReleaseTensorId(re),s._OrtReleaseTensor(X),Re})()),F.push([K,Pe,[],"cpu"])}else{let xe=Fr(K),be=new xe(Te);new Uint8Array(be.buffer,be.byteOffset,be.byteLength).set(s.HEAPU8.subarray(re,re+be.byteLength)),F.push([K,Pe,be,"cpu"])}}finally{s.stackRestore(_e),K==="string"&&re&&s._free(re),L||s._OrtReleaseTensor(X)}}g&&!_&&(s._OrtClearBoundOutputs(g.handle)!==0&&he("Can't clear bound outputs."),ft.set(e,[p,c,f,g,_,!1]));for(let[q,X]of await Promise.all(R))F[q][2]=X;return Ct("wasm ProcessOutputTensor"),F}finally{(te=s.webnnOnRunEnd)==null||te.call(s,p),s.stackRestore(A),I.forEach(ee=>s._OrtReleaseTensor(ee)),T.forEach(ee=>s._OrtReleaseTensor(ee)),E.forEach(ee=>s._free(ee)),v!==0&&s._OrtReleaseRunOptions(v),$.forEach(ee=>s._free(ee))}},tn=e=>{let t=me(),r=ft.get(e);if(!r)throw new Error("invalid session id");let i=r[0],a=t._OrtEndProfiling(i);a===0&&he("Can't get an profile file name."),t._OrtFree(a)},rn=e=>{let t=[];for(let r of e){let i=r[2];!Array.isArray(i)&&"buffer"in i&&t.push(i.buffer)}return t}}),mt,Oe,Ut,ir,ar,Br,pa,Nr,St,kt,Od,of,uf,lf,df,pf,cf,hf,ff=P(()=>{Ue(),sf(),Bt(),Oa(),mt=()=>!!ye.wasm.proxy&&typeof document<"u",Ut=!1,ir=!1,ar=!1,Nr=new Map,St=(e,t)=>{let r=Nr.get(e);r?r.push(t):Nr.set(e,[t])},kt=()=>{if(Ut||!ir||ar||!Oe)throw new Error("worker not ready")},Od=e=>{switch(e.data.type){case"init-wasm":Ut=!1,e.data.err?(ar=!0,pa[1](e.data.err)):(ir=!0,pa[0]()),Br&&(URL.revokeObjectURL(Br),Br=void 0);break;case"init-ep":case"copy-from":case"create":case"release":case"run":case"end-profiling":{let t=Nr.get(e.data.type);e.data.err?t.shift()[1](e.data.err):t.shift()[0](e.data.out);break}}},of=async()=>{if(!ir){if(Ut)throw new Error("multiple calls to 'initWasm()' detected.");if(ar)throw new Error("previous call to 'initWasm()' failed.");if(Ut=!0,mt())return new Promise((e,t)=>{Oe==null||Oe.terminate(),ap().then(([r,i])=>{try{Oe=i,Oe.onerror=n=>t(n),Oe.onmessage=Od,pa=[e,t];let a={type:"init-wasm",in:ye};!a.in.wasm.wasmPaths&&(r||ha)&&(a.in.wasm.wasmPaths={wasm:new URL(""+new URL("ort-wasm-simd-threaded.jsep-Bvhpdk4G.wasm",import.meta.url).href,import.meta.url).href}),Oe.postMessage(a),Br=r}catch(a){t(a)}},t)});try{await Ra(ye.wasm),await Qa(ye),ir=!0}catch(e){throw ar=!0,e}finally{Ut=!1}}},uf=async e=>{if(mt())return kt(),new Promise((t,r)=>{St("init-ep",[t,r]);let i={type:"init-ep",in:{epName:e,env:ye}};Oe.postMessage(i)});await Ya(ye,e)},lf=async e=>mt()?(kt(),new Promise((t,r)=>{St("copy-from",[t,r]);let i={type:"copy-from",in:{buffer:e}};Oe.postMessage(i,[e.buffer])})):Hr(e),df=async(e,t)=>{if(mt()){if(t!=null&&t.preferredOutputLocation)throw new Error('session option "preferredOutputLocation" is not supported for proxy.');return kt(),new Promise((r,i)=>{St("create",[r,i]);let a={type:"create",in:{model:e,options:{...t}}},n=[];e instanceof Uint8Array&&n.push(e.buffer),Oe.postMessage(a,n)})}else return Xa(e,t)},pf=async e=>{if(mt())return kt(),new Promise((t,r)=>{St("release",[t,r]);let i={type:"release",in:e};Oe.postMessage(i)});Ja(e)},cf=async(e,t,r,i,a,n)=>{if(mt()){if(r.some(s=>s[3]!=="cpu"))throw new Error("input tensor on GPU is not supported for proxy.");if(a.some(s=>s))throw new Error("pre-allocated output tensor is not supported for proxy.");return kt(),new Promise((s,u)=>{St("run",[s,u]);let d=r,p={type:"run",in:{sessionId:e,inputIndices:t,inputs:d,outputIndices:i,options:n}};Oe.postMessage(p,rn(d))})}else return en(e,t,r,i,a,n)},hf=async e=>{if(mt())return kt(),new Promise((t,r)=>{St("end-profiling",[t,r]);let i={type:"end-profiling",in:e};Oe.postMessage(i)});tn(e)}}),ca,Rd,mf,G0=P(()=>{Ue(),ff(),J(),Aa(),up(),ca=(e,t)=>{switch(e.location){case"cpu":return[e.type,e.dims,e.data,"cpu"];case"gpu-buffer":return[e.type,e.dims,{gpuBuffer:e.gpuBuffer},"gpu-buffer"];case"ml-tensor":return[e.type,e.dims,{mlTensor:e.mlTensor},"ml-tensor"];default:throw new Error(`invalid data location: ${e.location} for ${t()}`)}},Rd=e=>{switch(e[3]){case"cpu":return new et(e[0],e[2],e[1]);case"gpu-buffer":{let t=e[0];if(!Na(t))throw new Error(`not supported data type: ${t} for deserializing GPU tensor`);let{gpuBuffer:r,download:i,dispose:a}=e[2];return et.fromGpuBuffer(r,{dataType:t,dims:e[1],download:i,dispose:a})}case"ml-tensor":{let t=e[0];if(!Da(t))throw new Error(`not supported data type: ${t} for deserializing MLTensor tensor`);let{mlTensor:r,download:i,dispose:a}=e[2];return et.fromMLTensor(r,{dataType:t,dims:e[1],download:i,dispose:a})}default:throw new Error(`invalid data location: ${e[3]}`)}},mf=class{async fetchModelAndCopyToWasmMemory(e){return lf(await Ma(e))}async loadModel(e,t){tt();let r;typeof e=="string"?r=await this.fetchModelAndCopyToWasmMemory(e):r=e,[this.sessionId,this.inputNames,this.outputNames,this.inputMetadata,this.outputMetadata]=await df(r,t),je()}async dispose(){return pf(this.sessionId)}async run(e,t,r){tt();let i=[],a=[];Object.entries(e).forEach(f=>{let g=f[0],_=f[1],w=this.inputNames.indexOf(g);if(w===-1)throw new Error(`invalid input '${g}'`);i.push(_),a.push(w)});let n=[],s=[];Object.entries(t).forEach(f=>{let g=f[0],_=f[1],w=this.outputNames.indexOf(g);if(w===-1)throw new Error(`invalid output '${g}'`);n.push(_),s.push(w)});let u=i.map((f,g)=>ca(f,()=>`input "${this.inputNames[a[g]]}"`)),d=n.map((f,g)=>f?ca(f,()=>`output "${this.outputNames[s[g]]}"`):null),p=await cf(this.sessionId,a,u,s,d,r),c={};for(let f=0;f<p.length;f++)c[this.outputNames[s[f]]]=n[f]??Rd(p[f]);return je(),c}startProfiling(){}endProfiling(){hf(this.sessionId)}}}),gf={};Vt(gf,{OnnxruntimeWebAssemblyBackend:()=>Ea,initializeFlags:()=>Ia,wasmBackend:()=>yf});var Ia,Ea,yf,H0=P(()=>{Ue(),ff(),G0(),Ia=()=>{(typeof ye.wasm.initTimeout!="number"||ye.wasm.initTimeout<0)&&(ye.wasm.initTimeout=0);let e=ye.wasm.simd;if(typeof e!="boolean"&&e!==void 0&&e!=="fixed"&&e!=="relaxed"&&(console.warn(`Property "env.wasm.simd" is set to unknown value "${e}". Reset it to \`false\` and ignore SIMD feature checking.`),ye.wasm.simd=!1),typeof ye.wasm.proxy!="boolean"&&(ye.wasm.proxy=!1),typeof ye.wasm.trace!="boolean"&&(ye.wasm.trace=!1),typeof ye.wasm.numThreads!="number"||!Number.isInteger(ye.wasm.numThreads)||ye.wasm.numThreads<=0)if(typeof self<"u"&&!self.crossOriginIsolated)ye.wasm.numThreads=1;else{let t=typeof navigator>"u"?Cg("node:os").cpus().length:navigator.hardwareConcurrency;ye.wasm.numThreads=Math.min(4,Math.ceil((t||1)/2))}},Ea=class{async init(e){Ia(),await of(),await uf(e)}async createInferenceSessionHandler(e,t){let r=new mf;return await r.loadModel(e,t),r}},yf=new Ea});Ue();Ue();Ue();var F0="1.23.0-dev.20250731-e753643480";{let e=(H0(),dr(gf)).wasmBackend;Pt("webgpu",e,5),Pt("webnn",e,5),Pt("cpu",e,10),Pt("wasm",e,10)}Object.defineProperty(ye.versions,"web",{value:F0,enumerable:!0});/**
* @license
* Copyright 2021 Google LLC. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* =============================================================================
*//**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 *//**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */export{ye as _,et as q,Xd as w};
