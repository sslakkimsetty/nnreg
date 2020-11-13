library("EBImage")


.valid_img <- function(img) {
    cls <- class(img)

    # Check class of fixed and package
    if ( !cls == "Image" | !attr(cls, "package") == "EBImage" ) {
        stop(
            "Image is not of 'Image' class of EBImage package",
            call. = FALSE)}

    img
}


.validate_fixed_input <- function(fixed, scale_to, crop_lim) {
    cls <- class(fixed)

    fixed <- .valid_img(fixed)

    # Check intensity limits
    int <- as.array(fixed)
    if ( !all(!is.na(int)) | !all(int >= 0)) {
        stop(
            "Missing values or negative intensities found!",
            call. = FALSE
        )
    }

    # Check scaling factor limits
    if ( !all(scale_to >= 0) ) {
        stop(
            "Scaling factor or dimensions must be non-negative",
            call. = FALSE
        )
    }

    # Check cropping limits
    if ( !all(crop_lim >= 0) ) {
        stop(
            "Image cropping limits should be non-negative",
            call. = FALSE
        )
    }

    if ( all(crop_lim == 1) ) {
        stop(
            "Cropping in only a point", call. = FALSE
        )
    }

    if ( !(all(crop_lim >= 1) | all(crop_lim <= 1)) ) {
        stop(
            "Cropping should be either with percentage or pixels",
            call. = FALSE
        )
    }

    if ( !all(crop_lim[1:2] <= dim(fixed)[1]) | !all(crop_lim[3:4] <= dim(fixed)[2]) ) {
        stop(
            "Cropping limits should not exceed dimensions", call. = FALSE
        )
    }

    fixed
}


preprocess_fixed <- function(fixed,
    scale_to=c(1,1), scale_type="ratio", crop_lim=c(0,1,0,1)) {

    fixed <- .validate_fixed_input(fixed, scale_to, crop_lim)
    channels <- fixed@colormode + 1 # !!! check this

    dims <- dim(fixed)[1:2]

    # if (channels == 1) { # !!! fix this
    #     fixed <- as.array(fixed, dim = c(dims, 1))
    #     fixed <- Image(fixed)
    # }

    scale_type <- match.arg(scale_type, c("pixels", "ratio"))

    fixed <- .validate_fixed_input(fixed, scale_to, crop_lim)

    # Crop image
    if ( all(crop_lim >= 1) ) {
        if ( !all(is.integer(crop_lim)) & crop_lim < 256 ) {
            stop(
                "Pixel numbers should be integers!", call. = FALSE
            )
        }

        crop_lim <- c(crop_lim[1:2]/dims[1], crop_lim[3:4]/dims[2])
        print(crop_lim)
    }

    # Subset image
    w1 <- floor(crop_lim[1] * dims[1])
    w2 <- ceiling(crop_lim[2] * dims[1])
    h1 <- floor(crop_lim[3] * dims[2])
    h2 <- ceiling(crop_lim[4] * dims[2])

    if (channels == 1) {
        fixed <- fixed[w1:w2, h1:h2]
    } else {
        fixed <- fixed[w1:w2, h1:h2, ]
    }


    # Scale image
    if ( scale_type == "ratio" ) {
        scale_to = dims * scale_to
    }
    fixed <- resize(fixed, w=scale_to[1], h=scale_to[2])

    # Convert to gray scale
    if ( channels == 3 ) {
        fixed <- channel(fixed, "gray")
    }

    # Normalize data
    if ( !all(range(fixed) == c(0, 1)) ) {
        fixed <- normalize(fixed)
    }

    fixed
}


simImage <- function(w=100, h=100, mean=3, sd=5) {
    stopifnot(!is.integer(w) & !is.integer(h))

    colormat <- matrix(rep(c("red", "green", "blue"), w*h), w, h)
    x <- Image(colormat)
    # display(x, "raster", interpolate=FALSE)

    offset <- array(rnorm(w*h*3, mean=mean, sd=sd), c(w, h, 3))
    sim <- as.array(x) + offset
    sim <- rgbImage(sim[, , 1], sim[, , 2], sim[, , 3])

    sim
}

# sim <- simImage(150, 200)
# sim <- abs(sim)

# sim2 <- preprocess_fixed(sim, c(1.5,1), "ratio")
# display(sim2, "raster")

# optp <- preprocess_fixed(opt, c(1,1), "ratio")
# display(optp, "raster")

# optp <- preprocess_fixed(opt, c(1.5,1), "ratio", c(0,0.5,0,1))

# # temporary working variables
# fixed <- preprocess_fixed(opt, scale_to=dim(moving), scale_type="pixels")
