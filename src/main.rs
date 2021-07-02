use std::{collections::HashSet, env, fs::File, path::{PathBuf}};
use std::io::prelude::*;
use image::{ImageBuffer, Rgba, io::Reader as ImageReader};

/// Domain is the structure with HashSet of pixel coordinates pxs
/// and not-axially-aligned bounding box (nAABB) top left `cbbmin`
/// and bottom right `cbbmax` corner coordinates.
///
/// Домен - структура данных с сетом координат пикселей изображения `pxs` и
/// горизонтально расположенным охватывающим прямоугольником с координатами
/// верхнего левого `cbbmin` и нижнего правого `cbbmax` углов.
#[derive(Debug, Default)]
struct Domain {
    pxs: HashSet<[u32; 2]>,
    cbbmin: [u32; 2],
    cbbmax: [u32; 2],
}

// /// Max x * y distance from current pixel
// /// Максимальный квадрат расстояния от текущего пикселя, до пикселя в домене
// const MAX_DIST: f32 = 128.0;


fn main() {
    let args: Vec<String> = env::args().collect();
    println!("Image path: {}", args[1].clone());
    let img = ImageReader::open(args[1].clone())
    .unwrap().decode().unwrap().to_rgba8();

    let data = find_islands(img);
    // recalculate_domains(&mut data);
    write_yolo(args[1].clone(), data);
}

fn find_islands(img: ImageBuffer<Rgba<u8>, Vec<u8>>) -> Vec<Domain> {
    let dim = img.dimensions();
    let (w, h) = (dim.0, dim.1);
    let imgsize: usize = (w * h) as usize;
    let mut visited = HashSet::<[u32; 2]>::with_capacity(imgsize);
    // Maximum number of domains is much lower than number of pixels, in case
    // there are many domains, this HashMap will be extended when needed
    let mut domains = Vec::<Domain>::with_capacity(imgsize / 128);

    for (i, px) in img.pixels().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        if px[3] > 0u8 && !visited.contains(&[x, y]) {
            if check_bounds(x, y, w, h).iter().all(|&x| x == true) {
                proc_dom(x, y,  &mut visited,  &mut domains, &img, &dim);
            }
        }
    }

    domains
}

fn proc_dom(x: u32, y: u32, vis: &mut HashSet<[u32; 2]>,
doms: &mut Vec<Domain>, img: &ImageBuffer<Rgba<u8>, Vec<u8>>, dim: &(u32, u32)) {
    // Move around the island in cascade
    // Проходим по пикселям каскадом
    // let mut domain_processed = false;
    let mut domain_px = Vec::<[u32; 2]>::with_capacity(256);
    // let mut domain_px_s = HashSet::<[u32; 2]>::with_capacity(256);
    // let mut c = [0u32; 2];
    // let mut coordsf = [[0u32; 2]; 8];
    let w = dim.0;
    let h = dim.1;
    let cx = x;
    let cy = y;
    let cxl = cx - 1;
    let cxr = cx + 1;
    let cyu = cy - 1;
    let cyd = cy + 1;

    // Just allocate memory, no real check is needed
    let cond = check_bounds(cx, cy, w, h);
    // domain_px.push([x, y]);
    // domain_px_s.insert([x, y]);
    let co = [
        [cxl, cyu], [cx, cyu], [cxr, cyu],
        [cxl, cy],  [cx, cy],  [cxr, cy],
        [cxl, cyd], [cx, cyd], [cxr, cyd],
    ];

    let mut px_new = 0;
    let mut dom_size = 0;

    // Initial domain
    for [i, j] in co {
        if img.get_pixel(i, j)[3] > 0 {
            px_new += 1;
            domain_px.push([i, j]);
            vis.insert([i, j]);
        }
    }

    // Recursive domain filling
    while px_new > 0 {
        dom_size = domain_px.len();
        let px_new_t = px_new;
        px_new = 0;

        for a in dom_size - px_new_t..dom_size {
            let [i, j] = domain_px.get(a).unwrap();
            let coo = generate_coords(*i, *j, w, h);

            for [k, l] in coo {
                if !vis.contains(&[k, l]) {
                    if img.get_pixel(k, l)[3] > 0 {
                        px_new += 1;
                        domain_px.push([k, l]);
                        vis.insert([k, l]);
                    }
                }
            }
        }
    }

    let mut cmin = domain_px.iter().next().unwrap().clone();
    let mut cmax = cmin;
    let mut pxs_t = HashSet::<[u32; 2]>::with_capacity(domain_px.len());

    for i in domain_px.iter() {
        pxs_t.insert(*i);
        let (j, k) = (i[0], i[1]);
        if j < cmin[0] {cmin[0] = j;}
        if k < cmin[1] {cmin[1] = k;}
        if j > cmax[0] {cmax[0] = j;}
        if k > cmax[1] {cmax[1] = k;}
    }
    doms.push(Domain {
        pxs: pxs_t,
        cbbmin: cmin,
        cbbmax: cmax,
    })
}

fn check_bounds(x: u32, y: u32, w: u32, h: u32) -> [bool; 4] {
    let c0 = x > 0;
    let c1 = y > 0;
    let c2 = x + 1 < w;
    let c3 = y + 1 < h;

    return [c0, c1, c2, c3];
}

fn generate_coords(x: u32, y: u32, w: u32, h: u32) -> Vec<[u32; 2]> {
    let mut coords = Vec::<[u32; 2]>::with_capacity(8);
    let check = check_bounds(x, y, w, h);

    match check {
        // Most of pixels
        [true, true, true, true] => {
            let xl = x - 1;
            let xr = x + 1;
            let yu = y - 1;
            let yd = y + 1;

            coords = vec![
                [xl, yu], [x, yu], [xr, yu],
                [xl, y],           [xr, y],
                [xl, yd], [x, yd], [xr, yd],
            ];
        },
        // Left edge
        [false, true, true, true] => {
            let xr = x + 1;
            let yu = y - 1;
            let yd = y + 1;

            coords = vec![
                [x, yu], [xr, yu],
                         [xr, y],
                [x, yd], [xr, yd],
            ];
        },
        // Top edge
        [true, false, true, true] => {
            let xl = x - 1;
            let xr = x + 1;
            let yd = y + 1;

            coords = vec![
                [xl, y],           [xr, y],
                [xl, yd], [x, yd], [xr, yd],
            ];

        },
        // Right edge
        [true, true, false, true] => {
            let xl = x - 1;
            let yu = y - 1;
            let yd = y + 1;

            coords = vec![
                [xl, yu], [x, yu],
                [xl, y],
                [xl, yd], [x, yd],
            ];
        },
        // Bottom edge
        [true, true, true, false] => {
            let xl = x - 1;
            let xr = x + 1;
            let yu = y - 1;

            coords = vec![
                [xl, yu], [x, yu], [xr, yu],
                [xl, y],           [xr, y],
            ];
        },
        // Top left corner
        [false, false, true, true] => {
            let xr = x + 1;
            let yd = y + 1;

            coords = vec![
                         [xr, y],
                [x, yd], [xr, yd],
            ];
        },
        // Bottom right corner
        [true, true, false, false] => {
            let xl = x - 1;
            let yu = y - 1;

            coords = vec![
                [xl, yu], [x, yu],
                [xl, y],
            ];
        },
        _ => {panic!("Pixel coordinate is incorrect!")}
    }
    return coords;
}

fn recalculate_domains(doms: &mut Vec<Domain>) {
    let dom_num = doms.len();
    let mut doms_updated = Vec::<Domain>::with_capacity(dom_num / 16);
    let mut doms_processed = HashSet::<usize>::with_capacity(dom_num);

    for (i, d0) in doms.iter().rev().enumerate() {
        if !doms_processed.contains(&i) {
            let d0c = [
                (d0.cbbmax[0] - d0.cbbmin[0]) / 2,
                (d0.cbbmax[1] - d0.cbbmin[1]) / 2,
            ];

            for (j, d1) in doms.iter().rev().enumerate() {
                if i == j || doms_processed.contains(&j) {continue;} else {
                    let d1c = [
                        (d0.cbbmax[0] - d0.cbbmin[0]) / 2,
                        (d0.cbbmax[1] - d0.cbbmin[1]) / 2,
                    ];
                }
            }
        }
    }
    //     let (mut cbbmin, mut cbbmax) = (d.cbbmin, d.cbbmax);
    //     for [j, k] in &d.pxs {
    //         if j < &cbbmin[0] {cbbmin[0] = *j;}
    //         if k < &cbbmin[1] {cbbmin[1] = *k;}
    //         if j > &cbbmax[0] {cbbmax[0] = *j;}
    //         if k > &cbbmax[1] {cbbmax[1] = *k;}
    //     }
    //     d.cbbmin = cbbmin;
    //     d.cbbmax = cbbmax;
    // }
}

fn write_yolo(img: String, data: Vec<Domain>) {
    let mut path = PathBuf::from(img);
    path.set_extension("yolo");
    let path = path.as_path();
    let display = path.display();
    let mut fl = match File::create(path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };

    for i in &data {
        println!("Debug: {:?}", i);
    }

    let dataw: String = data.into_iter().enumerate()
        .map(|(i, a)| format!(
            "{} {}\n", i, format!(
                "{} {} {} {}",
                (a.cbbmax[0] - a.cbbmin[0]) / 2,
                (a.cbbmax[1] - a.cbbmin[1]) / 2,
                a.cbbmax[0] - a.cbbmin[0],
                a.cbbmax[1] - a.cbbmin[1],
                )
            )
        ).collect();

    match fl.write_all(dataw.as_bytes()) {
        Err(why) => panic!("Couldn't write to {}: {}", display, why),
        Ok(_) => println!("Successfully wrote to {}", display),
    }
}