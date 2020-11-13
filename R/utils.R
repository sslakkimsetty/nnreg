library("EBImage")
library("Cardinal")


.validate_overlay_images <- function(fixed, moving, alpha) {
    cls1 <- class(fixed)
    cls2 <- class(moving)
    if ( !cls1 == "Image" | !attr(cls1, "package") == "EBImage" |
         !cls2 == "Image" | !attr(cls2, "package") == "EBImage") {
        stop("Fixed and/or Moving images is not of 'Image' class
             of EBImage package", call.=FALSE)
    }

    if (alpha < 0 | alpha > 1) {
        stop("alpha has to be in [0,1] included", call.=FALSE)
    }
    fixed
}


overlay_images <- function(fixed, moving, alpha=0.25, new=FALSE) {
    .validate_overlay_images(fixed, moving, alpha)

    if (new){
        quartz()
    }
    display((alpha*fixed + (1-alpha)*moving), method="raster")
}



.validate_interpolate <- function(img, coords, method) {
    method <- match.arg(method, c("bilinear", "nearest_neighbor"))

    # img <- .valid_img(img)

    x <- dim(img)[1]
    y <- dim(img)[2]

    if (coords[1] <= 0 | coords[1] > x |
        coords[2] <= 0 | coords[2] > y) {
        stop("Coords out of bounds", call.=FALSE)}

    TRUE
}


interpolate <- function(img, coords=c(h=1,w=1), method="bilinear") {
    .valid <- .validate_interpolate(img, coords, method)

    h <- coords[[1]]
    w <- coords[[2]]

    h1 <- floor(h)
    h2 <- ceiling(h)
    w1 <- floor(w)
    w2 <- ceiling(w)

    neighbors <- matrix(c(h1,w1, h1,w2, h2,w1, h2,w2), ncol=2, byrow=TRUE)
    int <- apply(neighbors, 1, function(x) img[x[1], x[2]])

    if (method == "nearest_neighbor") {
        d <- apply(neighbors, 1, function(x) {
            dist(matrix(c(x, coords), ncol=2, byrow=TRUE), method="euclidean")
        })
        # sum(int * d) / sum(d)
        int[which.min(d)]

    } else if (method == "bilinear") {
        if (w1 == w2) {
            k1 <- int[1]
            k2 <- int[2]
        } else {
            k1 <- (w2 - w) * int[1] + (w - w1) * int[2]
            k2 <- (w2 - w) * int[3] + (w - w1) * int[4]
        }

        if (h1 == h2) {
            k <- k1
        } else {
            k <- (h2 - h) * k1 + (h - h1) * k2
        }
        k
    }

}
