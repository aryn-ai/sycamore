import React, { useRef } from "react";
import {
  ActionIcon,
  Anchor,
  AppShell,
  Box,
  Button,
  Center,
  Container,
  Flex,
  Group,
  Header,
  Loader,
  Popover,
  SimpleGrid,
  Stack,
  Text,
  useMantineTheme,
} from "@mantine/core";
import { useLocation, useSearchParams } from "react-router-dom";
import { useState, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { StyleSheet } from "@react-pdf/renderer";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "pdfjs-dist/build/pdf.worker.entry";
import {
  IconCircleArrowLeft,
  IconCircleArrowRight,
  IconCircleMinus,
  IconCirclePlus,
  IconInfoCircle,
  IconMinus,
  IconPlus,
  IconZoomIn,
  IconZoomOut,
} from "@tabler/icons-react";
import "./PdfViewer.css";
import { useDisclosure } from "@mantine/hooks";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.js",
  import.meta.url,
).toString();

function isValid(left: string, top: string, ranges = []) {
  const l = parseFloat(left.replace("%", "")) * 0.01;
  const t = parseFloat(top.replace("%", "")) * 0.01;
  for (const range of ranges) {
    console.log(range[0], range[1], range[2], range[3]);
    if (range[0] < l && l < range[1] && range[2] < t && t < range[3]) {
      return true;
    }
  }
  return false;
}

const fetchPDFThroughProxy = async (url: string) => {
  try {
    console.log("Getting pdf for: ", url);
    const response = await fetch("/v1/pdf", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url: url,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const pdfBlob = await response.blob();
    return URL.createObjectURL(pdfBlob);
  } catch (error) {
    console.error("Error fetching PDF through proxy:", error);
    throw error;
  }
};

export default function PdfViewer() {
  const [numPages, setNumPages] = useState<number>(-1);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [url, setUrl] = useState("");
  const [originalUrl, setOriginalUrl] = useState("");
  const [title, setTitle] = useState("Untitled");
  const [entity, setEntity] = useState<{ [key: string]: string }>({});
  const [boxes, setBoxes] = useState<any>();
  const [loading, setLoading] = useState(true);
  const [infoOpened, setInfoOpened] = useState(false);
  const theme = useMantineTheme();
  const [scale, setScale] = useState(1.5);
  const initialized = useRef(false);

  useEffect(() => {
    const loadPdf = async () => {
      console.log("Loading metadata");
      const dataString: string =
        localStorage.getItem("pdfDocumentMetadata") ?? "";
      const pdfDocumentMetadata: any = JSON.parse(dataString);
      const props: any = {};
      props["Document id"] = pdfDocumentMetadata.id;
      setTitle(pdfDocumentMetadata.title);
      const pdfURl = pdfDocumentMetadata.url
        ? pdfDocumentMetadata.url
        : pdfDocumentMetadata.properties.path;
      setOriginalUrl(pdfDocumentMetadata.properties.path);
      const response = await fetchPDFThroughProxy(pdfURl);
      setUrl(response);

      let pageNum: number = 1;

      if (pdfDocumentMetadata.properties.hasOwnProperty("page_number")) {
        pageNum = pdfDocumentMetadata.properties.page_number;
      } else if (
        pdfDocumentMetadata.properties.hasOwnProperty("page_numbers") &&
        Array.isArray(pdfDocumentMetadata.properties.page_numbers)
      ) {
        pageNum = pdfDocumentMetadata.properties.page_numbers[0];
      } else if (pdfDocumentMetadata.properties.hasOwnProperty("box")) {
        pageNum = pdfDocumentMetadata.properties.box;
      } else if (
        pdfDocumentMetadata.properties.hasOwnProperty("boxes") &&
        Array.isArray(pdfDocumentMetadata.properties.boxes)
      ) {
        pageNum = pdfDocumentMetadata.properties.boxes[0];
      }

      let boxObj: any = null;
      const bbox: number[] = pdfDocumentMetadata.bbox;
      if (bbox) {
        // This is overly complex.  We can simplify when we
        // drop the backward compatibility stuff below.
        boxObj = { [pageNum]: [bbox] };
      } else {
        if ("boxes" in pdfDocumentMetadata.properties)
          boxObj = pdfDocumentMetadata.properties.boxes;
        else if ("coordinates" in pdfDocumentMetadata.properties)
          boxObj = pdfDocumentMetadata.properties.points;
      }
      if ("entity" in pdfDocumentMetadata.properties) {
        Object.keys(pdfDocumentMetadata.properties.entity).forEach((key) => {
          props[key] = pdfDocumentMetadata.properties.entity[key];
        });
      }
      if (boxObj) {
        setBoxes(boxObj);
        let firstPage = Math.min(...Object.keys(boxObj).map(Number));
        if (!firstPage) {
          firstPage = pageNum;
          if (!firstPage) {
            firstPage = 1;
          }
        }
        setPageNumber(firstPage);
      } else {
        setPageNumber(pageNum); // i guess
      }
      setEntity(props);
      console.log("pdfDocumentMetadata is ", pdfDocumentMetadata);
      setLoading(false);
    };
    if (!initialized.current) {
      initialized.current = true;
      loadPdf();
    }
  }, []);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }): void {
    setNumPages(numPages);
  }
  const goToPrevPage = () =>
    setPageNumber(pageNumber - 1 <= 1 ? 1 : pageNumber - 1);

  const goToNextPage = () =>
    setPageNumber(pageNumber + 1 >= numPages ? numPages : pageNumber + 1);

  const handleZoomIn = () => {
    setScale((prevScale) => prevScale + 0.2);
  };

  const handleZoomOut = () => {
    setScale((prevScale) => Math.max(0.2, prevScale - 0.2));
  };

  return (
    <AppShell
      header={
        <Header
          height={50}
          bg={theme.colors.gray[7]}
          style={{
            borderBottom: "1px solid darkgray",
          }}
        >
          <SimpleGrid cols={loading ? 1 : 3} h="100%">
            {!loading && (
              <Flex align="center" justify="center" h="100%">
                <Group>
                  <Group spacing="xs">
                    <ActionIcon
                      onClick={goToPrevPage}
                      sx={(theme) => ({
                        "&:hover, &:active": { backgroundColor: "transparent" },
                      })}
                    >
                      <IconCircleArrowLeft stroke={2} color="white" />
                    </ActionIcon>
                    <ActionIcon
                      onClick={goToNextPage}
                      sx={(theme) => ({
                        "&:hover, &:active": { backgroundColor: "transparent" },
                      })}
                    >
                      <IconCircleArrowRight stroke={2} color="white" />
                    </ActionIcon>
                  </Group>
                  <Text c="white">
                    {" "}
                    {pageNumber} / {numPages}
                  </Text>
                </Group>
              </Flex>
            )}
            <Flex align="center" justify="center" h="100%">
              <Text fw="600" ta="center" c="white">
                {title}
              </Text>
            </Flex>
            {!loading && (
              <Flex align="center" justify="center" h="100%">
                <Popover
                  position="bottom"
                  withArrow
                  shadow="md"
                  opened={infoOpened}
                  onChange={setInfoOpened}
                >
                  <Popover.Target>
                    <ActionIcon
                      onClick={() => setInfoOpened((o) => !o)}
                      sx={(theme) => ({
                        "&:hover, &:active": { backgroundColor: "transparent" },
                      })}
                    >
                      <IconInfoCircle stroke={2} color="white" />
                    </ActionIcon>
                  </Popover.Target>
                  <Popover.Dropdown w="fit-content">
                    <Anchor fz="xs" href={originalUrl} target="_blank">
                      {originalUrl}
                    </Anchor>
                    {Object.entries(entity).map(([key, value]) => (
                      <Group>
                        <Text fw={700} fz="xs">
                          {key}:{" "}
                        </Text>
                        <Text fz="xs">{value}</Text>
                      </Group>
                    ))}
                  </Popover.Dropdown>
                </Popover>
              </Flex>
            )}
          </SimpleGrid>
        </Header>
      }
      styles={() => ({
        main: { padding: 0, paddingTop: 50, backgroundColor: "lightgray" },
      })}
    >
      {loading ? (
        <Flex justify="center" align="center" h="100%">
          <Loader />
        </Flex>
      ) : (
        <>
          <Document
            file={url}
            onLoadSuccess={onDocumentLoadSuccess}
            className="document"
          >
            <Page pageNumber={pageNumber} scale={scale} className="page">
              {boxes
                ? boxes[pageNumber] &&
                  boxes[pageNumber].map((box: any, index: number) => (
                    <div
                      key={index}
                      style={{
                        position: "absolute",
                        backgroundColor: "#ffff0033",
                        left: `${box[0] * 100}%`,
                        top: `${box[1] * 100}%`,
                        width: `${(box[2] - box[0]) * 100}%`,
                        height: `${(box[3] - box[1]) * 100}%`,
                      }}
                    />
                  ))
                : ""}
            </Page>
          </Document>

          <Stack
            pos="fixed"
            bottom="3rem"
            right="1.5rem"
            spacing={2}
            style={{ borderRadius: 5, zIndex: 100 }}
          >
            <ActionIcon
              onClick={handleZoomIn}
              sx={(theme) => ({
                border: "1px solid darkgray",
                backgroundColor: "white",
                "&:hover, &:active": { backgroundColor: "white" },
              })}
            >
              <IconZoomIn stroke={2} />
            </ActionIcon>
            <ActionIcon
              onClick={handleZoomOut}
              sx={(theme) => ({
                border: "1px solid darkgray",
                backgroundColor: "white",
                "&:hover, &:active": { backgroundColor: "white" },
              })}
            >
              <IconZoomOut stroke={2} />
            </ActionIcon>
          </Stack>
        </>
      )}
    </AppShell>
  );
}
