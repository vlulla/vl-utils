##
## Here are some variables and functions that I use regularly in my work.
##
## To use this in your interactive session put this somewhere in your .Rprofile file
##
## ## -- begin copy
## .my_env <- new.env()
## if(file.exists("~/R/utils.R")) { source("~/R/utils.R", local=.my_env)) }
## attach(.my_env)
## ### -- end copy
##
## Trying to insure that your interactive/session environment is *NOT* clobbered!
##
## Some of the more useful functions (in order of my daily usage) in here are:
##   lsos, toggleError, ispkginstalled, theme_vl, allfreqs, freqsdt,
##   run_examples_from_package, getAllS3methods, symdiff
##   genrandstr, genrandfilename
##
## Multiple names for functions is just to make interactive use easier...

trimspaces <- base::trimws
classes <- function(x) paste(class(x), collapse=", ")
replacespaces <- function(x, pattern="[\t\n ]", replace="_", ...) {
  gsub(pattern, replace, x, ...)
}
reverse <- function(str, split="") {
  ## See ?strsplit
  sapply(lapply(strsplit(str, split), rev), paste, collapse="")
}

day.name <- day.names <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
day.abb <- substr(day.name, 1, 3)
month.days <- c(31,28,31,30,31,30,31,31,30,31,30,31)
month.leap.days <- c(31,29,31,30,31,30,31,31,30,31,30,31)
alphanum <- ALPHANUM <- c(letters, LETTERS, 0:9)

ispkginstalled <- function(pkgname) {
  ischar <- tryCatch(is.character(pkgname) && length(pkgname) == 1L,
                     error=identity)
  if(inherits(ischar, "error")) ischar <- FALSE
  if(!ischar) pkgname <- deparse(substitute(pkgname))
  pkgname %in% names(utils::installed.packages()[,"Package"])
}

updatepkgs <- my.update.packages <- function(...) {
  local({r <- getOption("repos")
    r["CRAN"] <- "https://cloud.r-project.org"
    r["INLA"] <- "https://inla.r-inla-download.org/R/stable"
    options(repos=r)})
  update.packages(checkBuilt=TRUE, ask=FALSE, dependencies=TRUE)
}

# std. error
std_error <- std.error <- se <- function(x) sd(x)/sqrt(length(x))
symdiff <- function(x,y) union(setdiff(x,y), setdiff(y,x))

deg2rad <- function(d) d * pi / 180
rad2deg <- function(r) r * 180 / pi
# https://stackoverflow.com/a/26757297
cart2pol <- function(x,y) {
  rho <- sqrt(x^2+y^2)
  phi <- atan2(y,x)
  ## list(ρ=ρ,θ=θ) ## symbols are hard to type in R console!
  list(rho=rho,phi=phi)
}
## pol2cart <- function(ρ,θ) { list(x=ρ*cos(θ), y=ρ*sin(θ)) }
pol2cart <- function(rho,phi) { list(x=rho*cos(phi), y=rho*sin(phi)) }

cov.pop <- function(x, y=NULL, ...) { cov(x, y, ...) * (NROW(x)-1)/NROW(x) }
var.pop <- function(x, ...) { var(x, ...) * (NROW(x)-1)/NROW(x) }
sd.pop <- function(x, na.rm=FALSE) { sqrt(var.pop(x, na.rm=na.rm)) }
rmse <- RMSE <- function(residuals) sqrt(mean(residuals))
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }

fixcolnames <- normalize_string <- function(x, lowercase=FALSE) {

  ## Very useful for GIS related work!!!!

  f <- ifelse(lowercase, tolower, identity)
  gsub("^([0-9]+)?_|_$", "", f(gsub("[^A-Za-z0-9]+", "_", as.character(x))))
}

num_unique <- nunique <- function(x) length(unique(x))
getpaths <- pathcomponents <- function(path=Sys.getenv("PATH")) {
  unlist(strsplit(path, .Platform$path.sep))
}

totitle <- function(x, USE.NAMES=FALSE) {
  s <- sapply(x, function(x) strsplit(x, "\\s", perl=TRUE, fixed=FALSE), USE.NAMES=USE.NAMES)
  s <- sapply(s, function(x) paste(gsub("(.)(.*)", "\\U\\1\\E\\2", x, perl=TRUE), collapse=" "), USE.NAMES=USE.NAMES)
  s
}

toggleError <- function() {
  invisible(ifelse(is.null(options()$error), options(error=utils::recover), options(error=NULL)))
}

withOptions <- function(optlist, expr) {
  ## See the section ``Deep End'' on the excellent http://www.burns-stat.com/the-options-mechanism-in-r/
  ##
  ## Use it like this:
  ##
  ## R> print((1:5)^-1)
  ## [1] 1.000000 0.5000000 0.3333333 0.2500000 0.2000000
  ##
  ## R> withOptions(list(digits=3), print((1:5)^-1))
  ## [1] 1.000 0.500 0.333 0.250 0.200
  ##
  oldopt <- options(optlist)
  on.exit(options(oldopt))
  expr <- substitute(expr)
  eval.parent(expr)
}

withPar <- withPars <- function(parlist, expr) {
  ## See the section ``Deep End'' on the excellent http://www.burns-stat.com/the-options-mechanism-in-r/
  ##
  ## R> plot(mtcars$mpg, mtcars$displ)
  ## R> withPar(list(mar=c(1,1,1,1),pch=16), plot(mtcars$mpg, mtcars$disp))
  oldpar <- par(parlist)
  on.exit(par(oldpar))
  expr <- substitute(expr)
  eval.parent(expr)
}

theme_vl <- theme_VL <- function(base_size=11L) {

  theme_bw(base_size=base_size) +
    theme(
          plot.title=element_text(hjust=0.5,size=rel(1.8), face="bold"),
          plot.subtitle=element_text(size=rel(1), lineheight=1.1, hjust=1),
          ## plot.caption.position='plot',
          ## plot.caption=element_text(size=rel(0.7), hjust=0.1,lineheight=1.1, color="#555555"),
          plot.margin=margin(1.5,1.5,0.5,0.5,unit="line"),
          axis.text.x=element_text(color="black", margin=margin(t=0.3,unit='cm')),
          axis.text.y=element_text(color="black"),
          axis.ticks.length.x=unit(-0.2,'cm'),
          axis.ticks.length.y=unit( 0.2,'cm'),
          panel.border=element_rect(fill=NA, color="black"),
          panel.grid.major=element_line(color='gray80', linewidth=rel(0.7)),
          panel.grid.minor=element_line(color='gray90', linewidth=rel(0.5), linetype='dotted'),
          panel.spacing=unit(1.25, 'lines'),
          legend.background=element_rect(fill=NA, color="grey20", linewidth=0.2),
          legend.key.width=unit(0.8,'line'),
          legend.text=element_text(size=rel(0.8)),
          legend.title=element_text(size=rel(0.9)),
          ## legend.title=element_blank(),
          legend.position="bottom",
          strip.background=element_blank(),
          validate=TRUE
          )
}

add_credits <- function() {

  ## R> p <- ggplot(iris, aes(x=Petal.Width,y=Sepal.Width, color=Species)) + geom_point()
  ## R> print(p)
  ## R> add_credits()
  ##
  ## NOTE: these interfere with plot.caption!!
  ##
  grid::grid.text("http://vlulla.github.io",x=0.01,y=0.015,just='left',
                  gp=grid::gpar(col='#888888', fontsize=10, fontface="bold"))
  grid::grid.text(strftime(Sys.time(), "Plotted %Y.%m.%d %H:%M"), x=0.99, y=0.015, just="right",
                  gp=grid::gpar(col='#888888', fontsize=10, fontface="bold"))
}

no_axis_titles <- function() { theme(axis.title=element_blank()) }

squote <- singlequote <- function(x) sprintf("'%s'", x)
dquote <- doublequote <- function(x) sprintf('"%s"', x)

groups <- function(vector, numitems, overlap=FALSE) {
  ## Emulates the following two J/APL idioms
  ##     3 ]\ i.10
  ##    _3 ]\ i.10
  ##
  ## Try: groups(1:10, 3) ## group of 3 items of non-overlapping subsequences
  ##      groups(1:10, 3, overlaps=T) ## overlapping sequences

  if(overlap) {
    startidx <- 1:(length(vector) - numitems + 1)
  } else {
    startidx <- seq(1, length(vector), by=numitems)
  }
  endidx <- startidx + numitems - 1
  t(mapply(function(x,y) vector[x:y], startidx, endidx))
}

prefixes <- function(x) {
  ## This is analogous to J's \
  ##
  ## In J try:
  ## ]\'banana'
  ##
  ## In R:
  ## R> prefixes(1:5)
  ## R> prefixes(letters[1:5])
  ## R> all.equal(unlist(lapply(prefixes(1:5),sum)), cumsum(1:5))
  ## R> all.equal(unlist(lapply(prefixes(1:5),prod)), cumprod(1:5))
  ##
  ## This can be useful in creating some cumulative functions...for e.g.
  ## cumsd <- function(x) {
  ##   stopifnot(is.vector(x), is.numeric(x) || is.integer(x))
  ##   unlist(lapply(prefixes(x), sd))
  ## }
  ## cummean <- function(x) {
  ##   stopifnot(is.vector(x), is.numeric(x) || is.integer(x))
  ##   unlist(lapply(prefixes(x), mean))
  ## }
  stopifnot(is.vector(x))
  mapply(function(a,b) x[seq.int(a,b)], rep(1,length(x)), seq_along(x))
}

suffixes <- function(x) {
  ## This is analogous to J's \.
  ##
  ## In J try:
  ## ]\. 'banana'
  ## banana
  ## anana
  ## nana
  ## ana
  ## na
  ## a
  ##
  ## I don't know how useful this will be in R.
  ## R> suffixes(unlist(strsplit('banana','')))
  ##
  stopifnot(is.vector(x))
  mapply(function(a,b) x[seq.int(a,b)], seq_along(x), rep(length(x),length(x)))
}

## Returns the length of a function
function.length <- function(f) {
  if(is.character(f))
    f <- match.fun(f)
  length(deparse(f))
}

package.functions <- function(package, all.names=FALSE) {
  qual_pkg <-  ## qualified package name!
    if(isTRUE(startsWith(package, "package:"))) {
      package
    } else {
      sprintf("package:%s", package)
    }
  bare_pkg <- gsub("^package:", "", qual_pkg)
  if (! qual_pkg %in% search()) {
    orig_search <- search()
    require(bare_pkg, character.only=TRUE)
    pkgs_added <- setdiff(search(), orig_search) ## pkgs added by our require call!
    for(p in pkgs_added) {
      on.exit(detach(name=p, character.only=TRUE),add=TRUE) ## ?`detach`
    }
  }
  obj.names <- ls(name=qual_pkg, all.names=all.names)
  objs <- lapply(obj.names, get, qual_pkg)
  names(objs) <- obj.names
  Filter(is.function, objs)
}

package.function.lengths <- function(package) {
  vapply(package.functions(package), function.length, 0L)
}

st_rook <- function(a, b=a) sf::st_relate(a, b, pattern="F***1****") # ?sf::st_relate
st_queen <- function(a, b=a) sf::st_relate(a, b, pattern="F***T****")

lsos <- lsobjs <- .ls.objects  <- function(pos=1L, pattern, order.by, decreasing=FALSE, head=FALSE, n=5) {

  ##
  ## See http://stackoverflow.com/questions/1358003/tricks-to-manage-the-available-memory-in-an-r-session
  ##
  ## Modified to sort correctly based on size! There's a subtle bug in the SO answer

  classes <- function(x) paste(class(x), collapse=", ")
  napply <- function(names, fn) sapply(names, function(x) fn(get(x, pos=pos)))
  names <- ls(pos=pos, pattern=pattern)
  if(length(names) == 0L) return(character(0))

  ## obj.class <- napply(names, function(x) as.character(class(x))[[1]])
  obj.class <- napply(names, classes)
  obj.mode <- napply(names, base::mode)
  obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
  obj.size <- napply(names, function(x) {
                       l <- capture.output(print(object.size(x), units="auto"))
                       l[length(l)]
                  })
  obj.dim <- t(napply(names, function(x) as.numeric(dim(x))[1:2]))
  vec <- is.na(obj.dim)[,1] & (obj.type != "function")
  obj.dim[vec, 1] <- napply(names, length)[vec]

  out <- data.frame(obj.type, obj.size, obj.dim)
  names(out) <- c("Type/Class", "Size", "Rows", "Columns")

  if(any(obj.type %in% c("RasterStack", "RasterBrick", "SpatRaster"))) {
    nlayers <- function(x) ifelse(inherits(x, c("RasterStack", "RasterBrick")),
                                  raster::nlayers(x),
                                  ifelse(inherits(x, c("SpatRaster")), terra::nlyr(x), NA))
    out <- cbind(out, Layers=napply(names, nlayers))
  }

  if(!missing(order.by)) {
    idx <- if (order.by=="Size") {
      sizes <- napply(names, object.size)
      order(sizes, decreasing=decreasing)
    } else {
      order(out[[order.by]], decreasing=decreasing)
    }
    out <- out[idx, ]
  }

  if(head)
    out <- head(out, n)

  gc() ## This function uses a lot of memory! Free it before exiting.
  return(as.data.table(out, keep.rownames="ID"))
}

numna <- numNA <- numnas <- numNAs <- function(x) sum(is.na(x))

## Column Details for a Data Frame
## ^^^    ^^^^^^^       ^    ^
colDetails <- function(DF) {
  stopifnot("Needs a data.frame or data.table" = inherits(DF, "data.frame"))
  colnames <- colnames(DF)
  colclasses <- sapply(DF, classes)
  colidx <- seq_along(DF)
  num_nas <- sapply(DF, numna)
  num_uniq <- sapply(DF, num_unique)
  DFM <- copy(DF)
  for(i in seq_len(ncol(DF))) {
    if(class(DF[[i]]) %in% c("factor","character")) { DFM[[i]] <- as.numeric(rep(NA,nrow(DFM))) }
  }
  colstats <- function(x, na.rm=TRUE) {
    ## idea from McElreath's rethinking::precis function
    stats <- c(mean(x,na.rm=na.rm), sd(x,na.rm=na.rm), quantile(x,probs=c(0.055,0.945)))
    names(stats) <- c("mean","sd","5.5%","94.5%")
    stats
  }
  stats <- t(apply(DF,2,colstats))
  histosparks <- sapply(DFM, histospark) ## see below for histospark
  DD <- data.table(ColName=colnames, ColClasses=colclasses, ColIdx=colidx, NumNA=num_nas, PctNA=round(100*num_nas/nrow(DF),3), NumUniq=num_uniq, PctUniq=round(100*num_uniq/nrow(DF),3), row.names=NULL)
  DD <- cbind(DD, stats)
  DD[,Histogram:=histosparks]
  DD[]
}

issorted <- function(x) all(order(x) == seq_along(x))

#### list_functions_at_pos <- function(pos, all.names=FALSE) {
####   package.functions(search()[pos], all.names=all.names)
#### }
####
#### list_functionargs_at_pos <- function(pos=-1L, all.names=FALSE) {
####   funcs <- list_functions_at_pos(pos, all.names=all.names)
####   lapply(funcs, args)
####
#### }
#### ## All of this hassle for what??  Just use lsf.str(pos)!!!!  # ?`utils::lsf.str`
#### ## Or better yet ls.str(pos, mode="function")

genrandstr <- function(stringlen=5L) {

  ## generate random string of length stringlen
  ## Try
  ## R> genrandstr(8)
  ## R> replicate(8, genrandstr(8))
  ##
  ## Some useful generating functions:
  ## R> strs <- replicate(500, genrandstr(10))  ## 500 strings of length 10
  ## R> rstrs <- sapply(sample(5:20, 500, replace=TRUE), genrandstr) ###  SUPER USEFUL!!! 500 strings of length between 5-20!! You need this...

  paste0(sample(c(letters,LETTERS), stringlen, replace=TRUE), collapse="")
}

run_examples_from_package <- function(pkgname, local=TRUE) {

  ## A very useful function!!
  ## Try
  ## R> run_examples_from_package("data.table", local=FALSE) ## data.table does something funky with environments....very hard to get it to work
  ## R> run_examples_from_package("sf") ## great example to see what it can do...
  ## R> run_examples_from_package("ggplot2") ## to see why we all love ggplot2!!!
  ## R> run_examples_from_package("grid") ## the package that made ggplot2/lattice possible!!!!!!!
  ##
  ## This will create two files in `getcwd()`. The output of plotting commands
  ## are in <pkgname>_examples.pdf and the output/message text are in
  ## <pkgname>_example_messages.txt files!
  ##
  ## The function tries to be careful not to overwrite files if they're
  ## already present. Best to run this in an empty directory...
  ## also helps if some examples create some other files.....

  qual_pkg <- if(startsWith(pkgname, "package:")) {
    pkgname
  } else {
    sprintf("package:%s", pkgname)
  }
  bare_pkg <- gsub("^package:","",qual_pkg)
  if (! qual_pkg %in% search()) {
    require(bare_pkg, character.only=TRUE)
    on.exit(detach(qual_pkg, character.only=TRUE)) ## ?`detach`
  }
  pdfout <- sprintf("%s_examples.pdf", bare_pkg)
  msgout <- sprintf("%s_examples_message.txt", bare_pkg)
  stopifnot("Would overwrite existing files!" = !file.exists(pdfout))
  stopifnot("Would overwrite existing files!" = !file.exists(msgout))

  pdf(pdfout); on.exit(dev.off(), add=TRUE)
  msg <- file(msgout, open="wt")
  sink(msg); sink(msg,type="message");on.exit(sink(), add=TRUE)
  op <- options(error=NULL, "example.ask"=FALSE); on.exit(options(op), add=TRUE)
  invisible(sapply(ls(qual_pkg), example, package=bare_pkg, character.only=TRUE, local=local))
}

poisson_binomial <- function(theta) {

  ## Poisson-binomial distribution
  ##   Return vector of probabilities for all outcomes in increasing order of count.
  ##   theta: vector of probability values in (0, 1)
  ##
  ## See http://discourse.mc-stan.org/t/poisson-binomial-distribution-any-existing-stan-implementation/4220/5
  ##

  N <- length(theta)
  if (N == 0) return(c(1));
  alpha <- matrix(-Inf, N + 1, N + 1);
  alpha[1, 1] <- 1;
  for (n in 1:N) {
    tot = 0;
    alpha[n + 1, tot + 1] = alpha[n, tot + 1] * (1 - theta[n]);

    if (n > 1) {
      for (tot in 1:(n - 1)) {
        alpha[n + 1, tot + 1] =
            alpha[n, tot] * theta[n] + alpha[n, tot  + 1] * (1 - theta[n]);
      }
    }

    tot = n;
    alpha[n + 1, tot + 1] = alpha[n, tot] * theta[n];
  }
  return(alpha[N + 1, 1:(N + 1)]);
}

## if base > 1 then higher values get assigned higher probabilities
## if 0 < base < 1 then lower values get assigned higher probabilities
softmax <- function(x,base=exp(1)) (base^x)/sum(base^x)

freqsdt <- freqsDT <- function(DT, groupcols, percent=TRUE) {

  ## Idea of freqsdt from https://st2.ning.com/topology/rest/1.0/file/get/4077505910?profile=original
  ## Modified for my preferences...
  ##
  ## This is how you use them:
  ## R> m <- as.data.table(mtcars)
  ## R> allfreqs(m) ## list freqs for all the columns....
  ## R> # But I'm only interested in cyl, gear, and am columns...so let's try them out...
  ## R> freqsdt(m, c("cyl", "gear", "am"))
  ## R> ### Hmm....this is not quite what I need... let's think of something....
  ## R> allfreqs(m[,.(cyl,gear,am)]) ## isn't this neat?
  ##
  ## R> i <- as.data.table(iris)
  ## R> allfreqs(i)
  ## R> freqsdt(i, c("Species"))

  stopifnot("needs a data.table" = is.data.table(DT),
            "column names not provided" = is.character(groupcols) & length(groupcols) > 0L,
            "not all groupcols in DT!" = all(groupcols %chin% names(DT)))
  is.categorical <- function(x) { is.logical(x) || is.factor(x) || is.character(x) }
  if(!all(sapply(DT[, ..groupcols], is.categorical))) {
    warning("Some of the grouping columns don't appear to be categorical")
  }
  res <- DT[, .(frequency=.N), by=groupcols][order(-frequency)][,percentage:=100*frequency/sum(frequency)]
  res ## To force it...???
  outcols <- colnames(res)
  if(!isTRUE(percent)) outcols <- setdiff(outcols, "percentage")
  res[, ..outcols]
}

## Also from https://st2.ning.com/topology/rest/1.0/file/get/4077505910?profile=original
allfreqs <- function(DT, catlim=100L) {
  stopifnot("needs a data.table" = is.data.table(DT),
            "Number of categories (as integer) not provided" = is.integer(catlim) & catlim > 0L)
  if(NROW(DT) > 1e6) {
    cat("###########################################################\n")
    cat("The datatable contains more than 1 million rows            \n")
    cat("  and this function crashes R easily...so subsetting       \n")
    cat("  datatable to only one million (randomly selected) rows...\n")
    cat("###########################################################\n\n\n")
    DT <- DT[sample(.N, 1e6), ]
  }
  names <- names(DT)
  namelen <- length(names)
  final <- data.table(NULL)
  for(i in seq_len(namelen)) {
    freqs <- freqsdt(DT, c(names[i]))
    if(nrow(freqs) <= catlim) {
      final <- rbind(final,
                     data.table(vname=names[i], value=as.factor(freqs[[1]]),
                                frequency=freqs[[2L]], percent=freqs[[3L]]))
    }
  }
  final
}

getAllS3methods <- function(func) {
  ## I was looking at ?sf::`st_cast` and wanted to see all the different st_cast methods that were defined for different geometry types.
  ## This is how I go about it...
  ## R> library("sf"); ?st_cast
  ## R> getAllS3methods("st_cast")
  ## R> str(getAllS3methods("st_cast"))
  stopifnot("need function name as a character string" = is.character(func),
            "more than one function provided" = length(func) == 1L)

  m <- .S3methods(func)
  ms <- gsub("\\*$", "", as.character(m)) ## remove '*' at end

  s3methods <- sapply(ms, function(x) c(strsplit(x, "\\.")))
  ## cannot use s3methods directly...
  ## table(lengths(s3methods))
  ## s3methods[lengths(s3methods) > 2L] ## print.data.table is one example of issue...

  fixed_s3methods <- lapply(s3methods, function(x) c(x[1], paste0(x[-1L], collapse=".")))
  funcdefs <- lapply(fixed_s3methods, function(x) do.call(getS3method, as.list(x)))
  funcdefs
}

generate_random_filename <- genrandfilename <-
  function(minlen=5L, maxlen=20L, filechars=paste0(c(letters,LETTERS,0:9),collapse=""), extensions=c('pdf','exe','txt','docx','xlsx','md','dat','csv','shp','prj','dbf'), allowspaces=FALSE) {
    stopifnot("needs integer minlen. Try suffix 'L' to coerce number to integer" = is.integer(minlen),
              "needs integer maxlen. Try suffix 'L' to coerce number to integer" = is.integer(maxlen),
              "minlen is negative?" = minlen > 0L,
              "maxlen is negative?" = maxlen > 0L,
              "maxlen is less than minlen?" = maxlen >= minlen)
    stopifnot("allowspaces needs to be logical" = isTRUE(allowspaces) || isFALSE(allowspaces),
              "character vector needed" = is.character(filechars),
              "character vector needed" = is.character(extensions))
    require("data.table");
    len <- sample(seq.int(minlen, maxlen),1)
    filechars <- unlist(strsplit(filechars, ""))
    char_prob_tbl <- data.table(char=filechars, prob=1/length(filechars))
    if(isTRUE(allowspaces)) {
      ## browser()
      char_prob_tbl <- rbindlist(list(char_prob_tbl, list(" ", 2/(length(filechars)+1))))
      char_prob_tbl[char != ' ', prob:=(1 - char_prob_tbl[char==' ', prob])/ .N]
      stopifnot("probability doesn't add to 1!!" = char_prob_tbl[, sum(prob)] == 1L)
    }
    ## filename <- paste0(sample(filechars, len, replace=T), collapse="")
    filename <- paste0(sample(char_prob_tbl$char, len, prob=char_prob_tbl$prob, replace=T), collapse="")
    ext <- sample(extensions, 1L, replace=TRUE) ## someone might just give one extension!
    filename <- sprintf("%s.%s", filename, ext)
    filename
  }

## from data.table/R/utils.R
isTRUEorFALSE <- function(x) is.logical(x) && length(x) == 1L && !is.na(x)
isTRUEorNA <- function(x) is.logical(x) && length(x) == 1L && (is.na(x) || x)

## Vandermonde matrix!
vander <- VM <- function(vec, powers) {
  stopifnot(is.vector(vec) && (is.integer(vec) || is.numeric(vec)))
  stopifnot(is.integer(powers))
  outer(vec, 0:powers, `^`)
}

permutations <- function(x, n=6L) {

  ## useful function to generate permutations of vector or data.frame.
  ##
  ## Can be used to generate random ordering so that you can check whether
  ## your algo/functions depend on particular ordering of variables.
  ##
  ## R> permutations(1:10)
  ## R> permutations(mtcars)

  stopifnot((is.vector(x) && length(x) > 1L) || (is.data.frame(x) && nrow(x) > 1L))

  idx <- seq_along(x)
  if(is.data.frame(x)) {
    idx <- seq_len(nrow(x))
  }

  ## indices <- replicate(n, sample(idx, length(idx))) ## Permutations are cols...
  ## indices <- t(replicate(n, sample(idx, length(idx)))) ## permutations are rows...
  ## indices <- lapply(seq_len(n), function(x) sample(idx, length(idx)))
  ##
  indices <- replicate(n, sample(idx, length(idx))) ## Permutations are cols...

  res <- if(is.vector(x)) {
    ## x[indices]
    lapply(seq_len(ncol(indices)), function(i) x[indices[,i]])
  } else {
    ## x[indices,,drop=FALSE] ## drop=FALSE...just in case we get 1 col data.frame!!
    lapply(seq_len(ncol(indices)), function(i) x[indices[, i], , drop=FALSE])
  }
  res
}

generate_random_date_range <- function(start_date=as.Date('2000-01-01'), end_date=Sys.Date(), num_days=30L) {
  ## Useful function to generate a date range based on start/end dates...
  ##
  ## R> generate_random_date_range(as.Date('2000-01-01'),as.Date('2022-12-31'), 30L)
  ## R> generate_random_date_range(as.Date('2000-01-01'),as.Date('2022-12-31'), as.integer(sample(365,1))) ## much more useful...imo
  ## R> generate_random_date_range(as.Date('2000-01-01'),as.Date('2022-12-31'), as.integer(sample(c(365,366),1,prob=c(3/4,1/4)))) ## even more interesting!
  ##
  stopifnot(class(start_date)==class(Sys.Date()), class(end_date)==class(Sys.Date()), is.integer(num_days),
     end_date > start_date, end_date - start_date >= num_days)
  with(list(start_dt=sample(seq(start_date, end_date - num_days, by='1 day'),1)),seq(start_dt, start_dt+num_days-1,by='1 day'))
}

gcp_to_DT <- function(qry, project, params=list()) {
  ## TODO (vijay): figure out how to run bq_perform_query. See https://github.com/r-dbi/bigrquery/blob/HEAD/R/bq-perform.R
  tb <- bq_project_query(project,qry) ## require('bigrquery')
  DT <- as.data.table(bq_table_download(tb,bigint="integer64")) ## ensure that you use `fill=bit64::as.integer64(NA)` in `dcast.data.table`!
  DT
}

## especially useful for install.packages("pkg",lib=lastElem(.libPaths()))
lastElem <- lastelem <- function(l)l[[length(l)]]

## Learned of this from Richard McElreath's book.
## Found this on https://github.com/hadley/precis/blob/master/R/histospark.R
histospark <- function(x, width=10L) {
  if(all(is.na(x))){return("")}
  sparks <- c("\u2581","\u2582","\u2583",'\u2585','\u2587')
  bins <- graphics::hist(x, breaks=width, plot=FALSE)
  factor <- cut(bins$counts / max(bins$counts), breaks=seq(0L,1L,length=length(sparks)+1L),labels=sparks,include.lowest=TRUE)
  paste0(factor,collapse="")
}
